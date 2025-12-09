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
    
    This mapping is based on semantic similarity between UCI features
    and Phish-Hook feature definitions.
    
    Args:
        df: DataFrame with UCI features
        
    Returns:
        DataFrame with F1-F8 features
    """
    # Create new DataFrame for Phish-Hook features
    phishhook_df = pd.DataFrame()
    
    # F1: Small Levenshtein Distance (Look-alike domain)
    # Map from Prefix_Suffix (suspicious prefix/suffix patterns)
    # -1 or 1 indicates suspicious patterns
    phishhook_df['F1'] = ((df['Prefix_Suffix'] == -1) | (df['Prefix_Suffix'] == 1)).astype(int)
    
    # F2: Deeply Nested Subdomains
    # Map from having_Sub_Domain
    # -1 indicates many subdomains (suspicious), 1 indicates few/none (legitimate)
    phishhook_df['F2'] = (df['having_Sub_Domain'] == -1).astype(int)
    
    # F3: Issued from Free CA
    # Map from SSLfinal_State (SSL certificate state)
    # -1 indicates suspicious SSL (could be free CA), 1 indicates legitimate
    phishhook_df['F3'] = (df['SSLfinal_State'] == -1).astype(int)
    
    # F4: Suspicious TLD
    # Map from Domain_registeration_length (short registration = suspicious TLD)
    # -1 indicates short registration (often suspicious TLDs)
    phishhook_df['F4'] = (df['Domain_registeration_length'] == -1).astype(int)
    
    # F5: Inner TLD in Subdomain
    # Map from having_Sub_Domain combined with Prefix_Suffix
    # Suspicious patterns often have inner TLDs
    phishhook_df['F5'] = ((df['having_Sub_Domain'] == -1) & 
                          ((df['Prefix_Suffix'] == -1) | (df['Prefix_Suffix'] == 1))).astype(int)
    
    # F6: Suspicious Keywords
    # Map from Request_URL, URL_of_Anchor, Links_in_tags
    # These features indicate suspicious content/keywords
    phishhook_df['F6'] = ((df['Request_URL'] == -1) | 
                          (df['URL_of_Anchor'] == -1) | 
                          (df['Links_in_tags'] == -1)).astype(int)
    
    # F7: High Shannon Entropy
    # Map from URL_Length (long URLs often have high entropy)
    # Also consider Abnormal_URL
    phishhook_df['F7'] = ((df['URL_Length'] == 1) & 
                          (df['Abnormal_URL'] == -1)).astype(int)
    
    # F8: Hyphens in Subdomain
    # Map from having_Sub_Domain and Prefix_Suffix
    # Hyphens often appear in suspicious subdomains
    phishhook_df['F8'] = ((df['having_Sub_Domain'] == -1) & 
                          (df['Prefix_Suffix'] == -1)).astype(int)
    
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

