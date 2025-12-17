"""
Campaign-level detection utilities

Groups certificate / URL events into campaigns using multiple heuristics
so we can detect multi-sample phishing campaigns (same IP, shortener, age bin,
similar URL structure, or similar brand target).

This module is conservative about external dependencies: domain age lookup
is attempted only if `whois` is installed; missing packages are tolerated
and domain_age will be None in that case.
"""
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import socket
import re

# Known URL shorteners (simple list)
SHORTENERS = {
    'bit.ly', 'tinyurl.com', 't.co', 'ow.ly', 'goo.gl', 'buff.ly', 'is.gd',
    'adf.ly', 'bitly.com', 'short.ly', 'lnkd.in'
}

try:
    import whois  # optional; used for domain age
except Exception:
    whois = None


def resolve_ips(domain: str) -> List[str]:
    """Resolve domain to list of IPv4 addresses (best-effort)."""
    try:
        # socket.gethostbyname_ex returns (hostname, aliaslist, ipaddrlist)
        _, _, ips = socket.gethostbyname_ex(domain)
        return ips
    except Exception:
        return []


def detect_shortener(domain: str) -> bool:
    """Return True if domain looks like a known shortener."""
    domain = domain.lower().strip()
    # strip subdomain prefixes like www
    domain = re.sub(r'^www\.', '', domain)
    parts = domain.split('.')
    if len(parts) < 2:
        return False
    base = '.'.join(parts[-2:])
    return base in SHORTENERS


def domain_age_days(domain: str) -> Optional[int]:
    """Attempt to get domain age in days using whois (optional).

    Returns None if whois is not available or lookup fails.
    """
    if whois is None:
        return None
    try:
        w = whois.whois(domain)
        creation = w.creation_date
        if creation is None:
            return None
        # creation_date may be a list
        if isinstance(creation, list):
            creation = creation[0]
        if not isinstance(creation, datetime):
            return None
        delta = datetime.utcnow() - creation
        return max(0, delta.days)
    except Exception:
        return None


def normalize_token(token: str) -> str:
    """Normalize a domain token for structure similarity checks."""
    # Lowercase, replace digits with 0, collapse repeated chars (simple)
    t = token.lower()
    t = re.sub(r'\d', '0', t)
    t = re.sub(r'(.)\1{2,}', r'\1\1', t)
    return t


class CampaignDetector:
    """Simple campaign-level detector.

    It collects events and groups them by multiple keys (ip, shortener,
    age bins, simple structure signature, brand target) and raises
    an alert when a campaign exceeds a configurable size.
    """

    def __init__(self, size_threshold: int = 3, time_window_seconds: int = 86400):
        self.size_threshold = size_threshold
        self.time_window = timedelta(seconds=time_window_seconds)
        # campaigns: key -> list of event dicts
        self.campaigns: Dict[str, List[Dict]] = defaultdict(list)

    def _structure_key(self, domain: str) -> str:
        parts = domain.split('.')
        leftmost = parts[0] if parts else domain
        token = normalize_token(leftmost)
        # signature: token length bucket + label count + hyphen count
        token_len = min(len(token), 20)
        labels = len(parts)
        hyphens = leftmost.count('-')
        return f"struct:{token_len}:{labels}:{hyphens}:{token[:6]}"

    def _age_bin(self, days: Optional[int]) -> Optional[str]:
        if days is None:
            return None
        if days < 30:
            return 'age:<30'
        if days < 180:
            return 'age:30-180'
        return 'age:>180'

    def _keys_for_domain(self, domain: str, ips: List[str], age_days: Optional[int]) -> List[str]:
        keys = []
        # IP keys
        for ip in ips:
            keys.append(f'ip:{ip}')
        # shortener key
        if detect_shortener(domain):
            keys.append('shortener')
        # age bin key
        age_key = self._age_bin(age_days)
        if age_key:
            keys.append(age_key)
        # structure key
        keys.append(self._structure_key(domain))
        return keys

    def add(self, cert_data: Dict, classification: Optional[Dict] = None) -> List[Dict]:
        """Add certificate data to campaigns.

        cert_data: expects at least keys: 'domains' (list of domains), 'timestamp' (iso str)
        classification: optional dict mapping domain -> prediction/prob

        Returns list of campaign alerts (possibly empty).
        """
        alerts = []
        timestamp = cert_data.get('timestamp')
        try:
            ts = datetime.fromisoformat(timestamp) if timestamp else datetime.utcnow()
        except Exception:
            ts = datetime.utcnow()

        for i, domain in enumerate(cert_data.get('domains', [])):
            # Best-effort resolution
            ips = cert_data.get('ips', [])
            # If ips is a map per domain, handle that
            if isinstance(ips, dict):
                domain_ips = ips.get(domain, [])
            else:
                domain_ips = ips

            # Age lookup may be provided, else None
            ages = cert_data.get('domain_ages', {})
            age_days = ages.get(domain) if isinstance(ages, dict) else None

            keys = self._keys_for_domain(domain, domain_ips, age_days)

            # classification info
            domain_result = None
            if classification and isinstance(classification, dict):
                domain_result = classification.get(domain)

            event = {
                'domain': domain,
                'timestamp': ts,
                'phishing_probability': (domain_result.get('phishing_probability')
                                        if domain_result else None),
                'prediction': (domain_result.get('prediction') if domain_result else None)
            }

            # add to all relevant campaign keys
            for key in keys:
                self.campaigns[key].append(event)

                # prune old items
                self._prune_campaign(key)

                # check threshold
                if len(self.campaigns[key]) >= self.size_threshold:
                    alert = self._summarize_campaign(key)
                    # attach key so caller knows why it triggered
                    alert['campaign_key'] = key
                    alert['trigger'] = 'size_threshold'
                    alerts.append(alert)

        return alerts

    def _prune_campaign(self, key: str):
        cutoff = datetime.utcnow() - self.time_window
        self.campaigns[key] = [e for e in self.campaigns[key] if e['timestamp'] >= cutoff]

    def _summarize_campaign(self, key: str) -> Dict:
        events = self.campaigns.get(key, [])
        domains = list({e['domain'] for e in events})
        size = len(events)
        phishing_probs = [e['phishing_probability'] for e in events if e['phishing_probability'] is not None]
        phishing_count = sum(1 for e in events if e.get('prediction') == 1)
        avg_prob = (sum(phishing_probs) / len(phishing_probs)) if phishing_probs else None
        last_seen = max(e['timestamp'] for e in events) if events else None

        return {
            'size': size,
            'unique_domains': domains,
            'phishing_count': phishing_count,
            'average_phishing_probability': avg_prob,
            'last_seen': last_seen.isoformat() if last_seen else None
        }


__all__ = [
    'CampaignDetector', 'resolve_ips', 'detect_shortener', 'domain_age_days'
]
