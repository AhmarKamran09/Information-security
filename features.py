"""
Phish-Hook Feature Extraction Module
Extracts F1-F8 features from domains as specified in the paper.
"""

import re
import math
from typing import List, Tuple
from Levenshtein import distance as levenshtein_distance


# Brand names and common phishing targets for F1
BRAND_KEYWORDS = [
    'google', 'gmail', 'microsoft', 'apple', 'amazon', 'facebook', 'twitter',
    'paypal', 'ebay', 'netflix', 'yahoo', 'linkedin', 'instagram', 'whatsapp',
    'dropbox', 'adobe', 'oracle', 'cisco', 'intel', 'nvidia', 'samsung',
    'bank', 'chase', 'wells', 'citi', 'boa', 'usbank', 'tdbank'
]


# Suspicious TLDs for F4
SUSPICIOUS_TLDS = [
    'ga', 'gdn', 'bid', 'kim', 'xyz', 'top', 'win', 'tk', 'ml', 'cf',
    'gq', 'men', 'work', 'online', 'click', 'download', 'stream', 'review'
]


# Inner TLDs for F5
INNER_TLDS = ['com', 'org', 'net', 'edu', 'gov']


# Suspicious keywords for F6
SUSPICIOUS_KEYWORDS = [
    'login', 'verify', 'update', 'secure', 'account', 'confirm', 'password',
    'bank', 'signin', 'sign-in', 'authenticate', 'validation', 'verify-account'
]


# Free CAs for F3
FREE_CAS = [
    "let's encrypt", "lets encrypt", "let's-encrypt", "lets-encrypt",
    "cpanel", "cloudflare", "zerossl", "zero ssl", "zero-ssl"
]


def extract_domain_parts(domain: str) -> Tuple[str, List[str], str]:
    """
    Extract domain parts: leftmost label, subdomains, TLD.
    
    Args:
        domain: Full domain string
        
    Returns:
        Tuple of (leftmost_label, subdomains_list, tld)
    """
    domain = domain.lower().strip()
    # Remove protocol if present
    domain = re.sub(r'^https?://', '', domain)
    domain = re.sub(r'^www\.', '', domain)
    domain = domain.split('/')[0]  # Remove path
    
    parts = domain.split('.')
    if len(parts) < 2:
        return domain, [], ''
    
    tld = parts[-1]
    leftmost = parts[0]
    subdomains = parts[1:-1] if len(parts) > 2 else []
    
    return leftmost, subdomains, tld


def f1_levenshtein_lookalike(domain: str, threshold: int = 2) -> int:
    """
    F1: Small Levenshtein Distance (Look-alike domain detection)
    
    Compares domain tokens against brand keywords.
    Returns 1 if minimum distance <= threshold, else 0.
    """
    leftmost, _, _ = extract_domain_parts(domain)
    
    min_distance = float('inf')
    for brand in BRAND_KEYWORDS:
        dist = levenshtein_distance(leftmost.lower(), brand.lower())
        min_distance = min(min_distance, dist)
    
    return 1 if min_distance <= threshold else 0


def f1_enhanced_brand_similarity(domain: str, use_embedding: bool = True, threshold: float = 0.5) -> int:
    """
    F1 Enhanced: Brand Similarity with Embedding (Improved version)
    
    Uses HYBRID approach: embedding + Levenshtein for best detection.
    Catches:
    - Visual similarity (g00gle, ɢoogle, gooogle) - via embedding
    - Phonetic similarity (paypol) - via embedding
    - Character substitutions (g00gle) - via Levenshtein
    - Token rearrangements - via embedding
    
    Uses OR logic: if EITHER method detects, return 1.
    
    Args:
        domain: Domain string to check
        use_embedding: Whether to use embedding-based detection
        threshold: Similarity threshold for embedding (0-1)
        
    Returns:
        1 if similar to brand (by either method), 0 otherwise
    """
    # Always check Levenshtein first (catches character substitutions)
    levenshtein_result = f1_levenshtein_lookalike(domain)
    
    if use_embedding:
        try:
            from brand_similarity import enhanced_f1_brand_similarity
            embedding_result = enhanced_f1_brand_similarity(domain, threshold)
            # Use OR logic: if either detects, return 1
            return 1 if (levenshtein_result == 1 or embedding_result == 1) else 0
        except ImportError:
            # Fallback to Levenshtein if module not available
            return levenshtein_result
        except Exception:
            # Fallback to Levenshtein on any error
            return levenshtein_result
    
    # Fallback to original Levenshtein method
    return levenshtein_result


def f2_deeply_nested_subdomains(domain: str, threshold: int = 3) -> int:
    """
    F2: Deeply Nested Subdomains
    
    Counts subdomain depth: number_of_labels - 2
    Returns 1 if subdomain_depth >= threshold, else 0.
    """
    parts = domain.lower().split('.')
    subdomain_depth = len(parts) - 2
    
    return 1 if subdomain_depth >= threshold else 0


def f3_free_ca(issuer: str) -> int:
    """
    F3: Issued from Free CA
    
    Checks if issuer contains free CA keywords.
    Returns 1 if issued from free CA, else 0.
    """
    if not issuer:
        return 0
    
    issuer_lower = issuer.lower()
    for ca in FREE_CAS:
        if ca in issuer_lower:
            return 1
    return 0


def f4_suspicious_tld(domain: str) -> int:
    """
    F4: Suspicious TLD
    
    Checks if TLD is in suspicious list.
    Returns 1 if suspicious TLD, else 0.
    """
    _, _, tld = extract_domain_parts(domain)
    return 1 if tld in SUSPICIOUS_TLDS else 0


def f5_inner_tld_in_subdomain(domain: str) -> int:
    """
    F5: Inner TLD in Subdomain
    
    Checks if any subdomain contains inner TLD keywords (com, org, net).
    Returns 1 if found, else 0.
    """
    _, subdomains, _ = extract_domain_parts(domain)
    
    for subdomain in subdomains:
        for inner_tld in INNER_TLDS:
            if inner_tld in subdomain.lower():
                return 1
    return 0


def f6_suspicious_keywords(domain: str) -> int:
    """
    F6: Suspicious Keywords
    
    Checks if domain contains suspicious keywords.
    Returns 1 if found, else 0.
    """
    domain_lower = domain.lower()
    for keyword in SUSPICIOUS_KEYWORDS:
        if keyword in domain_lower:
            return 1
    return 0


def shannon_entropy(text: str) -> float:
    """
    Calculate Shannon entropy of a string.
    """
    if not text:
        return 0.0
    
    entropy = 0.0
    text_lower = text.lower()
    for char in set(text_lower):
        p = text_lower.count(char) / len(text_lower)
        if p > 0:
            entropy -= p * math.log2(p)
    
    return entropy


def f7_high_entropy(domain: str, threshold: float = 3.5) -> int:
    """
    F7: High Shannon Entropy
    
    Computes entropy of leftmost label.
    Returns 1 if entropy >= threshold, else 0.
    """
    leftmost, _, _ = extract_domain_parts(domain)
    entropy = shannon_entropy(leftmost)
    
    return 1 if entropy >= threshold else 0


def f8_hyphens_in_subdomain(domain: str, threshold: int = 2) -> int:
    """
    F8: Hyphens in Subdomain
    
    Counts hyphens in subdomains.
    Returns 1 if hyphen_count >= threshold, else 0.
    """
    _, subdomains, _ = extract_domain_parts(domain)
    
    total_hyphens = 0
    for subdomain in subdomains:
        total_hyphens += subdomain.count('-')
    
    return 1 if total_hyphens >= threshold else 0


def extract_features(domain: str, issuer: str = "", use_enhanced_f1: bool = True) -> List[int]:
    """
    Extract all F1-F8 features from a domain.
    
    Args:
        domain: Domain string to analyze
        issuer: Certificate issuer (for F3)
        use_enhanced_f1: Whether to use enhanced brand similarity (embedding-based)
        
    Returns:
        List of 8 binary features [F1, F2, F3, F4, F5, F6, F7, F8]
    """
    # Use enhanced F1 if requested, otherwise use original Levenshtein
    f1_value = f1_enhanced_brand_similarity(domain, use_embedding=use_enhanced_f1) if use_enhanced_f1 else f1_levenshtein_lookalike(domain)
    
    features = [
        f1_value,
        f2_deeply_nested_subdomains(domain),
        f3_free_ca(issuer),
        f4_suspicious_tld(domain),
        f5_inner_tld_in_subdomain(domain),
        f6_suspicious_keywords(domain),
        f7_high_entropy(domain),
        f8_hyphens_in_subdomain(domain)
    ]
    
    return features


def extract_features_from_ct_entry(ct_entry: dict, use_enhanced_f1: bool = True) -> Tuple[List[int], List[str]]:
    """
    Extract features from a Certificate Transparency log entry.
    
    Args:
        ct_entry: CT log entry dictionary
        use_enhanced_f1: Whether to use enhanced brand similarity (embedding-based)
        
    Returns:
        Tuple of (features_list, domains_list)
    """
    try:
        data = ct_entry.get('data', {})
        leaf_cert = data.get('leaf_cert', {})
        domains = leaf_cert.get('all_domains', [])
        
        chain = data.get('chain', [])
        issuer = ""
        if chain and len(chain) > 0:
            issuer = chain[0].get('subject', {}).get('aggregated', '')
        
        # Extract features for each domain
        all_features = []
        for domain in domains:
            if domain:
                features = extract_features(domain, issuer, use_enhanced_f1=use_enhanced_f1)
                all_features.append(features)
        
        return all_features, domains
    except Exception as e:
        print(f"Error extracting features from CT entry: {e}")
        return [], []

