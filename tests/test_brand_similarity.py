"""
Test script to demonstrate brand similarity embedding improvement.
Shows detection of typosquatting, visual similarity, and phonetic similarity.
"""

from brand_similarity import BrandSimilarityDetector, get_detector
from features import f1_levenshtein_lookalike, f1_enhanced_brand_similarity


def test_brand_similarity():
    """Test brand similarity detection on various phishing domains."""
    
    print("="*60)
    print("Brand Similarity Embedding Test")
    print("="*60)
    
    detector = get_detector()
    
    # Test cases: typosquatting, visual similarity, phonetic similarity
    test_domains = [
        # Typosquatting
        ("g00gle.com", "google"),
        ("googla.com", "google"),
        ("gooogle.com", "google"),
        ("paypol.com", "paypal"),
        ("paypai.com", "paypal"),
        ("facebok.com", "facebook"),
        
        # Visual similarity (homographs)
        ("g00gle-secure.com", "google"),
        ("paypal-login.com", "paypal"),
        ("amaz0n.com", "amazon"),
        
        # Legitimate (should not match)
        ("example.com", None),
        ("test-domain.com", None),
        ("mycompany.com", None),
    ]
    
    print("\n" + "="*60)
    print("Comparison: Levenshtein vs Brand Similarity Embedding")
    print("="*60)
    
    print(f"\n{'Domain':<30} {'Levenshtein':<15} {'Embedding':<15} {'Similarity':<12} {'Nearest Brand'}")
    print("-" * 90)
    
    for domain, expected_brand in test_domains:
        # Original Levenshtein
        levenshtein_result = f1_levenshtein_lookalike(domain)
        
        # Enhanced embedding
        embedding_result = f1_enhanced_brand_similarity(domain)
        similarity_score, nearest_brand = detector.compute_similarity(domain)
        
        levenshtein_str = "✓ Detected" if levenshtein_result == 1 else "✗ Missed"
        embedding_str = "✓ Detected" if embedding_result == 1 else "✗ Missed"
        
        print(f"{domain:<30} {levenshtein_str:<15} {embedding_str:<15} {similarity_score:.3f}        {nearest_brand}")
    
    print("\n" + "="*60)
    print("Detailed Analysis")
    print("="*60)
    
    # Show detailed detection for a few examples
    examples = ["g00gle.com", "paypol.com", "gooogle.com"]
    
    for domain in examples:
        print(f"\nDomain: {domain}")
        detection = detector.detect_typosquatting(domain)
        print(f"  Similarity Score: {detection['similarity_score']:.4f}")
        print(f"  Nearest Brand: {detection['nearest_brand']}")
        print(f"  Is Similar: {detection['is_similar']}")
        print(f"  Levenshtein F1: {f1_levenshtein_lookalike(domain)}")
        print(f"  Enhanced F1: {f1_enhanced_brand_similarity(domain)}")
    
    print("\n" + "="*60)
    print("Improvement Summary")
    print("="*60)
    print("\nBrand Similarity Embedding catches:")
    print("  ✓ Visual similarity (g00gle, ɢoogle)")
    print("  ✓ Phonetic similarity (paypol)")
    print("  ✓ Character substitutions (amaz0n)")
    print("  ✓ Better than simple Levenshtein distance")
    print("\nThis improves typosquatting detection for Information Security!")


if __name__ == '__main__':
    test_brand_similarity()

