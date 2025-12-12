"""
UCI Dataset to Phish-Hook F1-F8 Feature Mapper
Maps UCI phishing dataset features to Phish-Hook F1-F8 features.
"""

import pandas as pd
import numpy as np
from typing import List


def map_uci_to_phishhook_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map UCI dataset features to Phish-Hook F1-F8 features.
    
    Improved mapping using multiple UCI features in sophisticated combinations
    to better approximate the domain-level F1-F8 features.
    
    Args:
        df: DataFrame with UCI features
        
    Returns:
        DataFrame with F1-F8 features
    """
    # Create helper patterns for better feature combinations
    suspicious_pattern = (
        (df['Abnormal_URL'] == -1) |
        (df['Prefix_Suffix'] == -1) |
        (df['Request_URL'] == -1)
    )
    
    subdomain_suspicious = (
        (df['having_Sub_Domain'] == -1) |
        ((df['having_Sub_Domain'] == 0) & (df['URL_Length'] == 1))
    )
    
    # F1: Small Levenshtein Distance (Look-alike domain)
    # Look-alike domains have suspicious prefix/suffix and abnormal patterns
    phishhook_df = pd.DataFrame()
    phishhook_df['F1'] = (
        (df['Prefix_Suffix'] == -1) |
        ((df['Abnormal_URL'] == -1) & (df['Prefix_Suffix'] != 1)) |
        ((df['having_IP_Address'] == -1) & suspicious_pattern)
    ).astype(int)
    
    # F2: Deeply Nested Subdomains
    # Multiple subdomains indicate phishing
    phishhook_df['F2'] = (
        (df['having_Sub_Domain'] == -1) |
        ((df['having_Sub_Domain'] == 0) & (df['URL_Length'] == 1) & (df['double_slash_redirecting'] == 1))
    ).astype(int)
    
    # F3: Issued from Free CA
    # Free CAs are commonly used by phishers
    phishhook_df['F3'] = (
        (df['SSLfinal_State'] == -1) |
        ((df['age_of_domain'] == -1) & (df['SSLfinal_State'] != 1)) |
        ((df['Domain_registeration_length'] == -1) & (df['SSLfinal_State'] != 1))
    ).astype(int)
    
    # F4: Suspicious TLD
    # Short registration and new domains often use suspicious TLDs
    phishhook_df['F4'] = (
        (df['Domain_registeration_length'] == -1) |
        ((df['age_of_domain'] == -1) & (df['Domain_registeration_length'] != 1)) |
        ((df['DNSRecord'] == -1) & (df['Domain_registeration_length'] == -1))
    ).astype(int)
    
    # F5: Inner TLD in Subdomain
    # Fake TLDs in subdomain require subdomains + suspicious patterns
    phishhook_df['F5'] = (
        (subdomain_suspicious & (df['Prefix_Suffix'] == -1)) |
        ((df['having_Sub_Domain'] == -1) & (df['Abnormal_URL'] == -1)) |
        ((df['having_Sub_Domain'] == 0) & suspicious_pattern & (df['URL_Length'] == 1))
    ).astype(int)
    
    # F6: Suspicious Keywords
    # Keywords in URLs, anchors, and links
    phishhook_df['F6'] = (
        (df['Request_URL'] == -1) |
        (df['URL_of_Anchor'] == -1) |
        (df['Links_in_tags'] == -1) |
        ((df['Submitting_to_email'] == 1) & (df['Request_URL'] != 1)) |
        ((df['SFH'] == -1) & (df['Request_URL'] != 1))
    ).astype(int)
    
    # F7: High Shannon Entropy
    # Random domains have long URLs with abnormal patterns
    phishhook_df['F7'] = (
        ((df['URL_Length'] == 1) & (df['Abnormal_URL'] == -1)) |
        ((df['URL_Length'] == 1) & (df['having_IP_Address'] == -1) & suspicious_pattern) |
        ((df['Shortining_Service'] == 1) & (df['URL_Length'] == 1))
    ).astype(int)
    
    # F8: Hyphens in Subdomain
    # Phishing domains use hyphens in subdomains
    phishhook_df['F8'] = (
        (subdomain_suspicious & (df['Prefix_Suffix'] == -1)) |
        ((df['having_Sub_Domain'] == -1) & (df['Abnormal_URL'] == -1)) |
        ((df['having_Sub_Domain'] == -1) & (df['Request_URL'] == -1)) |
        ((df['having_Sub_Domain'] == 0) & (df['Prefix_Suffix'] == -1) & (df['URL_Length'] == 1))
    ).astype(int)
    
    return phishhook_df


def prepare_training_data(csv_path: str) -> tuple:
    """
    Load and prepare UCI dataset for training.
    
    Args:
        csv_path: Path to UCI phishing dataset CSV
        
    Returns:
        Tuple of (X_features, y_labels) where:
        - X_features: numpy array of F1-F8 features
        - y_labels: numpy array of binary labels (1 for phishing, 0 for legitimate)
    """
    # Load dataset
    df = pd.read_csv(csv_path)
    
    # Map to Phish-Hook features
    phishhook_df = map_uci_to_phishhook_features(df)
    
    # Extract features
    X = phishhook_df[['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8']].values
    
    # Extract labels
    # -1 = Phishing, 0 = Suspicious, +1 = Legitimate
    # Convert to binary: Phishing/Suspicious = 1, Legitimate = 0
    y = df['Result'].values
    y_binary = (y <= 0).astype(int)  # -1 and 0 -> 1 (phishing/suspicious), 1 -> 0 (legitimate)
    
    return X, y_binary


def get_class_distribution(y: np.ndarray) -> dict:
    """
    Get class distribution statistics.
    
    Args:
        y: Label array
        
    Returns:
        Dictionary with class counts and percentages
    """
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    
    distribution = {}
    for label, count in zip(unique, counts):
        distribution[int(label)] = {
            'count': int(count),
            'percentage': float(count / total * 100)
        }
    
    return distribution

