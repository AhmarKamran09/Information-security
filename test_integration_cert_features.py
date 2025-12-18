"""
Integration test for certificate security features with feature extraction.
Tests that F9-F11 features integrate properly with F1-F8.
"""

import unittest
from datetime import datetime, timedelta
from features import extract_features, extract_features_from_ct_entry


class TestFeatureIntegration(unittest.TestCase):
    """Test integration of certificate security features with main feature extraction"""
    
    def test_extract_features_with_cert_data(self):
        """Test extracting all 11 features with certificate data"""
        now = datetime.now()
        cert_data = {
            'all_domains': ['example.com'],
            'leaf_cert': {
                'subject': {
                    'aggregated': '/CN=example.com',
                    'CN': 'example.com'
                },
                'issuer': {
                    'aggregated': '/C=US/O=Let\'s Encrypt/CN=R3',
                    'CN': 'R3'
                },
                'not_before': (now - timedelta(days=30)).timestamp(),
                'not_after': (now + timedelta(days=60)).timestamp()
            }
        }
        
        features = extract_features(
            domain='example.com',
            issuer='Let\'s Encrypt',
            cert_data=cert_data,
            include_cert_features=True
        )
        
        # Should have 11 features (F1-F11)
        self.assertEqual(len(features), 11)
        
        # F1-F8 should be extracted normally
        # F9-F11 should be extracted from cert_data
        # For this normal domain, all should be 0
        self.assertEqual(features[8], 0)  # F9: SAN not suspicious
        self.assertEqual(features[9], 0)  # F10: Not self-signed
        self.assertEqual(features[10], 0)  # F11: Validity not suspicious
    
    def test_extract_features_without_cert_data(self):
        """Test extracting features without certificate data (8 features)"""
        features = extract_features(
            domain='example.com',
            issuer='Let\'s Encrypt',
            include_cert_features=False
        )
        
        # Should have only 8 features (F1-F8)
        self.assertEqual(len(features), 8)
    
    def test_extract_features_with_cert_features_but_no_data(self):
        """Test that F9-F11 are padded with zeros when cert_data is None"""
        features = extract_features(
            domain='example.com',
            issuer='Let\'s Encrypt',
            cert_data=None,
            include_cert_features=True
        )
        
        # Should have 11 features with F9-F11 as zeros
        self.assertEqual(len(features), 11)
        self.assertEqual(features[8], 0)  # F9 padded
        self.assertEqual(features[9], 0)  # F10 padded
        self.assertEqual(features[10], 0)  # F11 padded
    
    def test_suspicious_certificate_features(self):
        """Test that suspicious certificate triggers F9-F11"""
        now = datetime.now()
        cert_data = {
            'all_domains': ['paypal-login.com', 'amazon-verify.com'],
            'leaf_cert': {
                'subject': {
                    'aggregated': '/CN=phishing.com',
                    'CN': 'phishing.com'
                },
                'issuer': {
                    'aggregated': '/CN=phishing.com',
                    'CN': 'phishing.com'
                },
                'not_before': (now - timedelta(days=2)).timestamp(),
                'not_after': (now + timedelta(days=15)).timestamp()
            }
        }
        
        features = extract_features(
            domain='paypal-login.com',
            issuer='Self-Signed',
            cert_data=cert_data,
            include_cert_features=True
        )
        
        # Should have 11 features
        self.assertEqual(len(features), 11)
        
        # F9: Multiple brands = 1
        self.assertEqual(features[8], 1)
        # F10: Self-signed = 1
        self.assertEqual(features[9], 1)
        # F11: Short validity + recently issued brand domain = 1
        self.assertEqual(features[10], 1)
    
    def test_extract_features_from_ct_entry_with_cert_features(self):
        """Test extracting features from CT entry with certificate features"""
        now = datetime.now()
        ct_entry = {
            'data': {
                'leaf_cert': {
                    'all_domains': ['example.com', 'www.example.com'],
                    'subject': {
                        'aggregated': '/CN=example.com',
                        'CN': 'example.com'
                    },
                    'issuer': {
                        'aggregated': '/C=US/O=Let\'s Encrypt/CN=R3',
                        'CN': 'R3'
                    },
                    'not_before': (now - timedelta(days=30)).timestamp(),
                    'not_after': (now + timedelta(days=60)).timestamp()
                },
                'chain': [{
                    'subject': {
                        'aggregated': '/C=US/O=Let\'s Encrypt/CN=R3'
                    }
                }]
            }
        }
        
        features_list, domains = extract_features_from_ct_entry(
            ct_entry,
            include_cert_features=True
        )
        
        # Should extract features for 2 domains
        self.assertEqual(len(features_list), 2)
        self.assertEqual(len(domains), 2)
        
        # Each feature set should have 11 features
        for features in features_list:
            self.assertEqual(len(features), 11)
    
    def test_backward_compatibility(self):
        """Test that old code still works without cert_features parameter"""
        # This should work with default parameters
        features = extract_features(
            domain='example.com',
            issuer='Let\'s Encrypt'
        )
        
        # By default, include_cert_features=True, so should have 11 features
        # but cert_data=None, so F9-F11 will be padded with zeros
        self.assertEqual(len(features), 11)


if __name__ == '__main__':
    unittest.main()
