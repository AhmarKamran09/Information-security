"""
SSL/TLS Certificate Security Analysis Module
Implements F9-F12 features for enhanced phishing detection.
"""

import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from Levenshtein import distance as levenshtein_distance


# Brand keywords for SAN analysis (reuse from features.py)
BRAND_KEYWORDS = [
    'paypal', 'amazon', 'apple', 'microsoft', 'google', 'facebook', 'netflix',
    'instagram', 'twitter', 'linkedin', 'ebay', 'yahoo', 'adobe', 'dropbox',
    'bank', 'chase', 'wellsfargo', 'citi', 'bofa', 'usbank'
]


def extract_san_domains(cert_data: dict) -> List[str]:
    """
    Extract Subject Alternative Names (SAN) from certificate data.
    
    Args:
        cert_data: Certificate data dictionary from CertStream
        
    Returns:
        List of domain names from SAN field
    """
    # CertStream provides all_domains which includes SAN
    domains = cert_data.get('all_domains', [])
    
    # Also check if there's explicit SAN data in extensions
    # (CertStream usually flattens this into all_domains)
    if not domains:
        # Fallback: try to extract from leaf_cert if available
        leaf_cert = cert_data.get('leaf_cert', {})
        domains = leaf_cert.get('all_domains', [])
    
    # Clean and normalize domains
    cleaned_domains = []
    for domain in domains:
        # Remove wildcards and clean
        domain = domain.replace('*.', '').strip().lower()
        if domain and domain not in cleaned_domains:
            cleaned_domains.append(domain)
    
    return cleaned_domains


def detect_brand_in_domain(domain: str) -> Optional[str]:
    """
    Detect if a domain contains a brand keyword.
    
    Args:
        domain: Domain string
        
    Returns:
        Brand keyword if found, None otherwise
    """
    domain_lower = domain.lower()
    # Remove TLD for better matching
    domain_parts = domain_lower.split('.')
    main_part = '.'.join(domain_parts[:-1]) if len(domain_parts) > 1 else domain_lower
    
    for brand in BRAND_KEYWORDS:
        # Check exact match or with small Levenshtein distance
        if brand in main_part:
            return brand
        # Check for typosquatting (distance <= 2)
        for part in domain_parts:
            if levenshtein_distance(brand, part) <= 2:
                return brand
    
    return None


def f9_san_analysis(cert_data: dict) -> int:
    """
    F9: SAN (Subject Alternative Names) Analysis
    
    Detects suspicious patterns in certificate SAN field:
    - Multiple unrelated brand names in one certificate
    - Excessive number of domains (>10 = suspicious)
    - Mix of suspicious and legitimate-looking domains
    
    Args:
        cert_data: Certificate data dictionary
        
    Returns:
        1 if suspicious SAN pattern detected, 0 otherwise
    """
    domains = extract_san_domains(cert_data)
    
    # No domains or single domain = not suspicious
    if len(domains) <= 1:
        return 0
    
    # Check 1: Excessive number of domains
    if len(domains) > 10:
        return 1
    
    # Check 2: Multiple different brands in one certificate
    brands_found = set()
    for domain in domains:
        brand = detect_brand_in_domain(domain)
        if brand:
            brands_found.add(brand)
    
    # Multiple different brands = suspicious
    if len(brands_found) >= 2:
        return 1
    
    # Check 3: Mix of very different domain structures
    # (e.g., paypal.com and random-string-123.xyz)
    tlds = [domain.split('.')[-1] if '.' in domain else '' for domain in domains]
    unique_tlds = set(tlds)
    
    # More than 3 different TLDs in one cert = suspicious
    if len(unique_tlds) > 3:
        return 1
    
    return 0


def f10_self_signed(cert_data: dict) -> int:
    """
    F10: Self-Signed Certificate Detection
    
    Checks if certificate is self-signed (issuer == subject).
    Self-signed certificates are common in phishing sites.
    
    Args:
        cert_data: Certificate data dictionary
        
    Returns:
        1 if self-signed, 0 otherwise
    """
    # Extract subject and issuer from leaf_cert
    leaf_cert = cert_data.get('leaf_cert', {})
    
    subject = leaf_cert.get('subject', {})
    issuer = leaf_cert.get('issuer', {})
    
    # If no data available, assume not self-signed (conservative)
    if not subject or not issuer:
        return 0
    
    # Known CAs that should not be flagged as self-signed
    known_cas = ['let\'s encrypt', 'digicert', 'comodo', 'godaddy', 
                 'sectigo', 'globalsign', 'entrust', 'cloudflare']
    
    # Compare aggregated strings (most reliable)
    subject_str = subject.get('aggregated', '').strip()
    issuer_str = issuer.get('aggregated', '').strip()
    
    if subject_str and issuer_str and subject_str == issuer_str:
        # Check if it's a known CA certificate
        issuer_lower = issuer_str.lower()
        if any(ca in issuer_lower for ca in known_cas):
            return 0
        return 1
    
    # Fallback: compare CN (Common Name)
    subject_cn = subject.get('CN', '').strip()
    issuer_cn = issuer.get('CN', '').strip()
    
    if subject_cn and issuer_cn and subject_cn == issuer_cn:
        # Additional check: if it's a known CA, it's not self-signed
        issuer_lower = issuer_cn.lower()
        if any(ca in issuer_lower for ca in known_cas):
            return 0
        return 1
    
    return 0


def f11_validity_period(cert_data: dict) -> int:
    """
    F11: Validity Period Analysis
    
    Analyzes certificate validity period for suspicious patterns:
    - Very short validity (<30 days) - common in phishing
    - Very long validity (>825 days) - violates CA/Browser Forum baseline
    - Recently issued certificates (<7 days ago) for popular brands
    
    Args:
        cert_data: Certificate data dictionary
        
    Returns:
        1 if suspicious validity period, 0 otherwise
    """
    leaf_cert = cert_data.get('leaf_cert', {})
    
    # Extract validity timestamps
    not_before = leaf_cert.get('not_before')
    not_after = leaf_cert.get('not_after')
    
    # If no validity data, assume not suspicious (conservative)
    if not not_before or not not_after:
        return 0
    
    try:
        # Convert timestamps to datetime
        # CertStream provides Unix timestamps (seconds since epoch)
        if isinstance(not_before, (int, float)):
            not_before_dt = datetime.fromtimestamp(not_before)
        else:
            # Try parsing as ISO string
            not_before_dt = datetime.fromisoformat(str(not_before))
        
        if isinstance(not_after, (int, float)):
            not_after_dt = datetime.fromtimestamp(not_after)
        else:
            not_after_dt = datetime.fromisoformat(str(not_after))
        
        # Calculate validity period
        validity_period = not_after_dt - not_before_dt
        validity_days = validity_period.days
        
        # Check 1: Very short validity (<30 days)
        if validity_days < 30:
            return 1
        
        # Check 2: Very long validity (>825 days)
        # CA/Browser Forum baseline requires max 825 days since Sept 2020
        if validity_days > 825:
            return 1
        
        # Check 3: Recently issued certificate for brand domains
        now = datetime.now()
        cert_age = now - not_before_dt
        cert_age_days = cert_age.days
        
        # If certificate is very new (<7 days)
        if cert_age_days < 7:
            # Check if any domain contains a brand keyword
            domains = extract_san_domains(cert_data)
            for domain in domains:
                if detect_brand_in_domain(domain):
                    return 1
        
        return 0
        
    except (ValueError, TypeError, OSError) as e:
        # If timestamp parsing fails, assume not suspicious
        return 0

def extract_cert_chain(cert_data: dict) -> List[Dict]:
    """
    Extract certificate chain from CertStream data.
    
    Args:
        cert_data: Certificate data dictionary from CertStream
        
    Returns:
        List of certificate dictionaries in the chain
    """
    # CertStream provides chain in data.chain
    data = cert_data.get('data', {})
    chain = data.get('chain', [])
    
    # If no chain in data, try direct access
    if not chain:
        chain = cert_data.get('chain', [])
    
    return chain if chain else []


def issuer_matches_subject(cert: Dict, issuer_cert: Dict) -> bool:
    """
    Check if a certificate's issuer matches the subject of the next cert in chain.
    
    Args:
        cert: Certificate dictionary
        issuer_cert: Issuer certificate dictionary
        
    Returns:
        True if issuer matches subject, False otherwise
    """
    # Get issuer from current cert
    cert_issuer = cert.get('issuer', {})
    # Get subject from issuer cert
    issuer_subject = issuer_cert.get('subject', {})
    
    # Compare aggregated strings
    cert_issuer_str = cert_issuer.get('aggregated', '').strip()
    issuer_subject_str = issuer_subject.get('aggregated', '').strip()
    
    if cert_issuer_str and issuer_subject_str:
        return cert_issuer_str == issuer_subject_str
    
    # Fallback: compare CN
    cert_issuer_cn = cert_issuer.get('CN', '').strip()
    issuer_subject_cn = issuer_subject.get('CN', '').strip()
    
    if cert_issuer_cn and issuer_subject_cn:
        return cert_issuer_cn == issuer_subject_cn
    
    return False


def f12_chain_validation(cert_data: dict) -> int:
    """
    F12: Certificate Chain Validation
    
    Validates certificate chain structure to detect:
    - Broken chains (issuer/subject mismatch)
    - Incomplete chains (missing intermediate certificates)
    - Suspicious single-certificate chains
    
    Args:
        cert_data: Certificate data dictionary
        
    Returns:
        1 if suspicious chain detected, 0 otherwise
    """
    # Get leaf certificate first
    leaf_cert = cert_data.get('leaf_cert', {})
    if not leaf_cert:
        return 0
    
    # Check if certificate is self-signed
    leaf_subject = leaf_cert.get('subject', {})
    leaf_issuer = leaf_cert.get('issuer', {})
    
    subject_str = leaf_subject.get('aggregated', '').strip()
    issuer_str = leaf_issuer.get('aggregated', '').strip()
    
    is_self_signed = (subject_str and issuer_str and subject_str == issuer_str)
    
    # Extract chain
    chain = extract_cert_chain(cert_data)
    
    # No chain data: suspicious if not self-signed
    if not chain or len(chain) == 0:
        # If not self-signed but no chain = suspicious
        if not is_self_signed and subject_str and issuer_str:
            return 1
        # Self-signed with no chain is OK (caught by F10)
        return 0
    
    # Check 1: Very short chain (suspicious for non-self-signed)
    # Legitimate certs usually have: leaf -> intermediate -> root (at least 2 in chain)
    if len(chain) < 2 and not is_self_signed:
        return 1
    
    # Check 2: Validate chain integrity (issuer/subject matching)
    # First, check if leaf cert's issuer matches first chain cert's subject
    if len(chain) > 0:
        first_chain_subject = chain[0].get('subject', {})
        
        leaf_issuer_str = issuer_str
        first_chain_str = first_chain_subject.get('aggregated', '').strip()
        
        # If both exist and don't match = broken chain
        if leaf_issuer_str and first_chain_str and leaf_issuer_str != first_chain_str:
            return 1
    
    # Check 3: Validate chain continuity
    for i in range(len(chain) - 1):
        if not issuer_matches_subject(chain[i], chain[i + 1]):
            return 1  # Broken chain
    
    return 0



def extract_cert_security_features(cert_data: dict) -> List[int]:
    """
    Extract all certificate security features (F9-F12).
    
    Args:
        cert_data: Certificate data dictionary from CertStream
        
    Returns:
        List of 4 binary features [F9, F10, F11, F12]
    """
    f9 = f9_san_analysis(cert_data)
    f10 = f10_self_signed(cert_data)
    f11 = f11_validity_period(cert_data)
    f12 = f12_chain_validation(cert_data)
    
    return [f9, f10, f11, f12]


def analyze_certificate_security(cert_data: dict) -> Dict:
    """
    Comprehensive certificate security analysis with detailed results.
    
    Args:
        cert_data: Certificate data dictionary
        
    Returns:
        Dictionary with detailed analysis results
    """
    domains = extract_san_domains(cert_data)
    f9 = f9_san_analysis(cert_data)
    f10 = f10_self_signed(cert_data)
    f11 = f11_validity_period(cert_data)
    f12 = f12_chain_validation(cert_data)
    
    # Get validity info
    leaf_cert = cert_data.get('leaf_cert', {})
    not_before = leaf_cert.get('not_before')
    not_after = leaf_cert.get('not_after')
    
    validity_days = None
    cert_age_days = None
    
    if not_before and not_after:
        try:
            if isinstance(not_before, (int, float)):
                not_before_dt = datetime.fromtimestamp(not_before)
                not_after_dt = datetime.fromtimestamp(not_after)
            else:
                not_before_dt = datetime.fromisoformat(str(not_before))
                not_after_dt = datetime.fromisoformat(str(not_after))
            
            validity_days = (not_after_dt - not_before_dt).days
            cert_age_days = (datetime.now() - not_before_dt).days
        except:
            pass
    
    # Get chain info
    chain = extract_cert_chain(cert_data)
    chain_length = len(chain)
    
    return {
        'san_domains': domains,
        'san_count': len(domains),
        'f9_san_suspicious': bool(f9),
        'f10_self_signed': bool(f10),
        'f11_validity_suspicious': bool(f11),
        'f12_chain_suspicious': bool(f12),
        'validity_days': validity_days,
        'certificate_age_days': cert_age_days,
        'chain_length': chain_length,
        'features': [f9, f10, f11, f12]
    }


__all__ = [
    'extract_san_domains',
    'extract_cert_chain',
    'f9_san_analysis',
    'f10_self_signed',
    'f11_validity_period',
    'f12_chain_validation',
    'extract_cert_security_features',
    'analyze_certificate_security'
]
