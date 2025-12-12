"""
Brand Similarity Embedding Module
Improves F1 (Levenshtein Distance) with vector-based brand similarity detection.
Uses character n-grams and cosine similarity for better typosquatting detection.
"""

import re
import numpy as np
from typing import List, Tuple, Dict
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os


# Major brand names for phishing detection
BRAND_LIST = [
    'google', 'gmail', 'microsoft', 'apple', 'amazon', 'facebook', 'twitter',
    'paypal', 'ebay', 'netflix', 'yahoo', 'linkedin', 'instagram', 'whatsapp',
    'dropbox', 'adobe', 'oracle', 'cisco', 'intel', 'nvidia', 'samsung',
    'bank', 'chase', 'wells', 'citi', 'boa', 'usbank', 'tdbank',
    'visa', 'mastercard', 'americanexpress', 'discover'
]


class BrandSimilarityDetector:
    """
    Brand similarity detector using character n-gram embeddings.
    Improves upon simple Levenshtein distance by capturing:
    - Visual similarity (g00gle, ɢoogle, gooogle)
    - Phonetic similarity (paypol)
    - Token rearrangements (login-google-secure)
    """
    
    def __init__(self, brands: List[str] = None, ngram_range: Tuple[int, int] = (3, 5)):
        """
        Initialize brand similarity detector.
        
        Args:
            brands: List of brand names to protect
            ngram_range: Character n-gram range for embedding
        """
        self.brands = brands if brands else BRAND_LIST
        self.ngram_range = ngram_range
        self.vectorizer = None
        self.brand_embeddings = None
        self._build_embeddings()
    
    def _build_embeddings(self):
        """Build character n-gram embeddings for all brands."""
        # Create character n-gram vectorizer
        self.vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=self.ngram_range,
            lowercase=True,
            max_features=1000
        )
        
        # Fit on brand names
        self.brand_embeddings = self.vectorizer.fit_transform(self.brands)
    
    def extract_domain_token(self, domain: str) -> str:
        """
        Extract the main domain token (leftmost label).
        
        Args:
            domain: Full domain string
            
        Returns:
            Main domain token
        """
        domain = domain.lower().strip()
        # Remove protocol if present
        domain = re.sub(r'^https?://', '', domain)
        domain = re.sub(r'^www\.', '', domain)
        domain = domain.split('/')[0]  # Remove path
        
        parts = domain.split('.')
        if len(parts) < 2:
            return domain
        
        return parts[0]
    
    def compute_similarity(self, domain: str, threshold: float = 0.5) -> Tuple[float, str]:
        """
        Compute similarity between domain and nearest brand.
        
        Uses both embedding similarity and Levenshtein distance for better detection.
        
        Args:
            domain: Domain string to check
            threshold: Similarity threshold for detection
            
        Returns:
            Tuple of (max_similarity, nearest_brand)
        """
        # Extract main token
        token = self.extract_domain_token(domain)
        
        # Skip if token is too short
        if len(token) < 3:
            return 0.0, ""
        
        # Transform domain token to embedding
        try:
            domain_embedding = self.vectorizer.transform([token])
            
            # Compute cosine similarity with all brands
            similarities = cosine_similarity(domain_embedding, self.brand_embeddings)[0]
            
            # Also compute Levenshtein distances for hybrid approach
            # This catches cases like g00gle that embedding might miss
            from Levenshtein import distance as levenshtein_distance
            levenshtein_scores = []
            min_levenshtein_dist = float('inf')
            best_levenshtein_idx = 0
            
            for i, brand in enumerate(self.brands):
                dist = levenshtein_distance(token.lower(), brand.lower())
                
                # Track minimum distance for later use
                if dist < min_levenshtein_dist:
                    min_levenshtein_dist = dist
                    best_levenshtein_idx = i
                
                # Convert distance to similarity (inverse, normalized)
                # Small distance = high similarity
                max_len = max(len(token), len(brand))
                if max_len == 0:
                    lev_sim = 1.0
                else:
                    # Normalize: distance of 0-2 chars = high similarity
                    if dist <= 2:
                        lev_sim = 1.0 - (dist / 3.0)  # 0->1.0, 1->0.67, 2->0.33
                    else:
                        lev_sim = max(0.0, 1.0 - (dist / max_len))
                levenshtein_scores.append(lev_sim)
            
            # Combine embedding similarity and Levenshtein similarity
            # Use maximum of both (OR logic) - if either detects, mark as similar
            # This ensures we catch both embedding cases and Levenshtein cases
            levenshtein_array = np.array(levenshtein_scores)
            combined_similarities = np.maximum(similarities, levenshtein_array)
            
            # If Levenshtein distance is small (<= 2), ensure high similarity
            # This catches cases like g00gle (distance=1) that embedding might miss
            if min_levenshtein_dist <= 2:
                # Boost the similarity for the best matching brand
                combined_similarities[best_levenshtein_idx] = max(
                    combined_similarities[best_levenshtein_idx], 
                    0.7  # Ensure it's above threshold of 0.5
                )
            
            # Find maximum similarity
            max_sim_idx = np.argmax(combined_similarities)
            max_similarity = float(combined_similarities[max_sim_idx])
            nearest_brand = self.brands[max_sim_idx]
            
            return max_similarity, nearest_brand
        except Exception as e:
            # Fallback to 0 if embedding fails
            return 0.0, ""
    
    def is_brand_similar(self, domain: str, threshold: float = 0.5) -> bool:
        """
        Check if domain is similar to any brand.
        
        Uses hybrid approach: embedding + Levenshtein for better detection.
        
        Args:
            domain: Domain string to check
            threshold: Similarity threshold (lowered to 0.5 for better recall)
            
        Returns:
            True if similar to any brand
        """
        similarity, _ = self.compute_similarity(domain, threshold)
        return similarity >= threshold
    
    def get_similarity_score(self, domain: str) -> float:
        """
        Get similarity score (0-1) for domain.
        
        Args:
            domain: Domain string to check
            
        Returns:
            Similarity score (0-1)
        """
        similarity, _ = self.compute_similarity(domain)
        return similarity
    
    def detect_typosquatting(self, domain: str, threshold: float = 0.5) -> Dict:
        """
        Detect typosquatting with detailed information.
        
        Args:
            domain: Domain string to check
            threshold: Similarity threshold
            
        Returns:
            Dictionary with detection results
        """
        similarity, nearest_brand = self.compute_similarity(domain, threshold)
        token = self.extract_domain_token(domain)
        
        return {
            'is_similar': similarity >= threshold,
            'similarity_score': similarity,
            'nearest_brand': nearest_brand,
            'domain_token': token,
            'threshold': threshold
        }
    
    def save(self, filepath: str):
        """Save detector to file."""
        data = {
            'brands': self.brands,
            'ngram_range': self.ngram_range,
            'vectorizer': self.vectorizer,
            'brand_embeddings': self.brand_embeddings
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, filepath: str):
        """Load detector from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        detector = cls(brands=data['brands'], ngram_range=data['ngram_range'])
        detector.vectorizer = data['vectorizer']
        detector.brand_embeddings = data['brand_embeddings']
        return detector


# Global detector instance
_detector = None


def get_detector() -> BrandSimilarityDetector:
    """Get or create global brand similarity detector."""
    global _detector
    if _detector is None:
        _detector = BrandSimilarityDetector()
    return _detector


def enhanced_f1_brand_similarity(domain: str, threshold: float = 0.5) -> int:
    """
    Enhanced F1 feature using brand similarity embedding.
    Combines Levenshtein distance with vector-based similarity.
    
    Args:
        domain: Domain string to check
        threshold: Similarity threshold
        
    Returns:
        1 if similar to brand, 0 otherwise
    """
    detector = get_detector()
    return 1 if detector.is_brand_similar(domain, threshold) else 0


def get_brand_similarity_score(domain: str) -> float:
    """
    Get brand similarity score (0-1).
    
    Args:
        domain: Domain string
        
    Returns:
        Similarity score
    """
    detector = get_detector()
    return detector.get_similarity_score(domain)

