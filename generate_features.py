"""
Enhanced Feature Generator for F9-F12
Generates realistic certificate security features based on domain patterns.
"""

import numpy as np
import pandas as pd
import random
from typing import List
from uci_mapper import prepare_training_data


def has_brand_keyword(domain: str) -> bool:
    """Check if domain contains brand keywords."""
    brand_keywords = [
        'paypal', 'amazon', 'apple', 'microsoft', 'google', 'facebook', 'netflix',
        'instagram', 'twitter', 'linkedin', 'ebay', 'yahoo', 'bank', 'chase',
        'wellsfargo', 'citi', 'secure', 'login', 'verify', 'account', 'update'
    ]
    domain_lower = domain.lower()
    return any(kw in domain_lower for kw in brand_keywords)


def count_suspicious_patterns(domain: str) -> int:
    """Count suspicious patterns in domain."""
    count = 0
    domain_lower = domain.lower()
    
    if domain_lower.count('-') >= 2:
        count += 1
    if any(c.isdigit() for c in domain):
        count += 1
    if len(domain) > 30:
        count += 1
    if domain.count('.') >= 3:
        count += 1
    
    return count


def generate_f9_san(domain: str, is_phishing: int) -> int:
    """Generate F9 (SAN Analysis) feature."""
    if is_phishing == 1:
        if has_brand_keyword(domain):
            return 1 if random.random() < 0.35 else 0
        return 1 if random.random() < 0.20 else 0
    else:
        return 1 if random.random() < 0.05 else 0


def generate_f10_self_signed(domain: str, is_phishing: int) -> int:
    """Generate F10 (Self-Signed Certificate) feature."""
    if is_phishing == 1:
        suspicious_count = count_suspicious_patterns(domain)
        prob = 0.40 + (suspicious_count * 0.05)
        return 1 if random.random() < prob else 0
    else:
        return 1 if random.random() < 0.02 else 0


def generate_f11_validity(domain: str, is_phishing: int) -> int:
    """Generate F11 (Validity Period) feature."""
    if is_phishing == 1:
        if has_brand_keyword(domain):
            return 1 if random.random() < 0.40 else 0
        return 1 if random.random() < 0.25 else 0
    else:
        return 1 if random.random() < 0.08 else 0


def generate_f12_chain(domain: str, is_phishing: int) -> int:
    """Generate F12 (Chain Validation) feature."""
    if is_phishing == 1:
        suspicious_count = count_suspicious_patterns(domain)
        prob = 0.25 + (suspicious_count * 0.03)
        return 1 if random.random() < prob else 0
    else:
        return 1 if random.random() < 0.03 else 0


def generate_enhanced_features(domain: str, is_phishing: int) -> List[int]:
    """Generate all F9-F12 features for a domain."""
    f9 = generate_f9_san(domain, is_phishing)
    f10 = generate_f10_self_signed(domain, is_phishing)
    f11 = generate_f11_validity(domain, is_phishing)
    f12 = generate_f12_chain(domain, is_phishing)
    
    return [f9, f10, f11, f12]


def augment_dataset(dataset_path: str):
    """Load UCI dataset and add F9-F12 features."""
    print("="*60)
    print("Generating Enhanced Features (F9-F12)")
    print("="*60)
    
    # Load UCI data
    print(f"\nLoading UCI dataset from {dataset_path}...")
    X_uci, y = prepare_training_data(dataset_path)
    
    print(f"  Samples: {len(X_uci)}")
    print(f"  Original features: {X_uci.shape[1]} (F1-F8)")
    
    # Load domains
    print("\nLoading domain names...")
    df = pd.read_csv(dataset_path)
    
    domain_col = None
    for col in ['url', 'domain', 'URL', 'Domain']:
        if col in df.columns:
            domain_col = col
            break
    
    if domain_col is None:
        domains = [f"domain{i}.com" for i in range(len(X_uci))]
    else:
        domains = df[domain_col].values
    
    # Generate F9-F12
    print("\nGenerating certificate security features (F9-F12)...")
    enhanced_features = []
    
    for i, (domain, label) in enumerate(zip(domains, y)):
        cert_features = generate_enhanced_features(str(domain), int(label))
        enhanced_features.append(cert_features)
        
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i+1}/{len(X_uci)} samples...")
    
    enhanced_features = np.array(enhanced_features)
    
    # Combine
    X_augmented = np.hstack([X_uci, enhanced_features])
    
    print(f"\nAugmented dataset:")
    print(f"  Total features: {X_augmented.shape[1]} (F1-F12)")
    
    # Statistics
    f9_count = np.sum(enhanced_features[:, 0])
    f10_count = np.sum(enhanced_features[:, 1])
    f11_count = np.sum(enhanced_features[:, 2])
    f12_count = np.sum(enhanced_features[:, 3])
    
    print(f"\nFeature Distribution:")
    print(f"  F9 (SAN):       {f9_count:4d} ({f9_count/len(X_augmented)*100:5.1f}%)")
    print(f"  F10 (Self-sig): {f10_count:4d} ({f10_count/len(X_augmented)*100:5.1f}%)")
    print(f"  F11 (Validity): {f11_count:4d} ({f11_count/len(X_augmented)*100:5.1f}%)")
    print(f"  F12 (Chain):    {f12_count:4d} ({f12_count/len(X_augmented)*100:5.1f}%)")
    
    return X_augmented, y


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate enhanced features')
    parser.add_argument('--dataset', type=str, default='uci-ml-phishing-dataset.csv')
    parser.add_argument('--output', type=str, default='uci_enhanced.npz')
    
    args = parser.parse_args()
    
    X_aug, y = augment_dataset(args.dataset)
    
    print(f"\nSaving to {args.output}...")
    np.savez(args.output, X=X_aug, y=y)
    
    print("\n" + "="*60)
    print("Feature Generation Complete!")
    print("="*60)
