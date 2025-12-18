"""
Unit Tests for Certificate Security Features (F9-F12)
Tests SAN analysis, self-signed detection, validity period analysis, and chain validation.
"""

import unittest
from datetime import datetime, timedelta
from cert_security import (
    extract_san_domains,
    extract_cert_chain,
    f9_san_analysis,
    f10_self_signed,
    f11_validity_period,
    f12_chain_validation,
    extract_cert_security_features,
    analyze_certificate_security
)


class TestSANAnalysis(unittest.TestCase):
    """Test F9: SAN (Subject Alternative Names) Analysis"""
    
    def test_single_domain(self):
        """Single domain should not be suspicious"""
        cert_data = {
            'all_domains': ['example.com']
        }
        self.assertEqual(f9_san_analysis(cert_data), 0)
    
    def test_multiple_related_domains(self):
        """Multiple related domains should not be suspicious"""
        cert_data = {
            'all_domains': ['example.com', 'www.example.com', 'mail.example.com']
        }
        self.assertEqual(f9_san_analysis(cert_data), 0)
    
    def test_excessive_domains(self):
        """More than 10 domains should be suspicious"""
        cert_data = {
            'all_domains': [f'domain{i}.com' for i in range(15)]
        }
        self.assertEqual(f9_san_analysis(cert_data), 1)
    
    def test_multiple_brands(self):
        """Multiple different brands in one cert should be suspicious"""
        cert_data = {
            'all_domains': ['paypal-login.com', 'amazon-verify.com']
        }
        self.assertEqual(f9_san_analysis(cert_data), 1)
    
    def test_multiple_tlds(self):
        """More than 3 different TLDs should be suspicious"""
        cert_data = {
            'all_domains': [
                'example.com',
                'example.net',
                'example.org',
                'example.xyz',
                'example.top'
            ]
        }
        self.assertEqual(f9_san_analysis(cert_data), 1)
    
    def test_extract_san_domains(self):
        """Test SAN domain extraction"""
        cert_data = {
            'all_domains': ['*.example.com', 'example.com', 'www.example.com']
        }
        domains = extract_san_domains(cert_data)
        self.assertIn('example.com', domains)
        # Wildcard should be removed
        self.assertNotIn('*.example.com', domains)


class TestSelfSignedDetection(unittest.TestCase):
    """Test F10: Self-Signed Certificate Detection"""
    
    def test_self_signed_cert(self):
        """Self-signed certificate should be detected"""
        cert_data = {
            'leaf_cert': {
                'subject': {
                    'aggregated': '/CN=example.com',
                    'CN': 'example.com'
                },
                'issuer': {
                    'aggregated': '/CN=example.com',
                    'CN': 'example.com'
                }
            }
        }
        self.assertEqual(f10_self_signed(cert_data), 1)
    
    def test_ca_signed_cert(self):
        """CA-signed certificate should not be detected as self-signed"""
        cert_data = {
            'leaf_cert': {
                'subject': {
                    'aggregated': '/CN=example.com',
                    'CN': 'example.com'
                },
                'issuer': {
                    'aggregated': '/C=US/O=Let\'s Encrypt/CN=R3',
                    'CN': 'R3'
                }
            }
        }
        self.assertEqual(f10_self_signed(cert_data), 0)
    
    def test_known_ca_not_self_signed(self):
        """Known CA should not be flagged even if CN matches"""
        cert_data = {
            'leaf_cert': {
                'subject': {
                    'aggregated': '/CN=Let\'s Encrypt',
                    'CN': 'Let\'s Encrypt'
                },
                'issuer': {
                    'aggregated': '/CN=Let\'s Encrypt',
                    'CN': 'Let\'s Encrypt'
                }
            }
        }
        # This is a CA cert itself, should not be flagged
        self.assertEqual(f10_self_signed(cert_data), 0)
    
    def test_missing_cert_data(self):
        """Missing certificate data should return 0 (conservative)"""
        cert_data = {}
        self.assertEqual(f10_self_signed(cert_data), 0)


class TestValidityPeriodAnalysis(unittest.TestCase):
    """Test F11: Validity Period Analysis"""
    
    def test_short_validity(self):
        """Certificate with <30 days validity should be suspicious"""
        now = datetime.now()
        not_before = now.timestamp()
        not_after = (now + timedelta(days=20)).timestamp()
        
        cert_data = {
            'leaf_cert': {
                'not_before': not_before,
                'not_after': not_after
            }
        }
        self.assertEqual(f11_validity_period(cert_data), 1)
    
    def test_long_validity(self):
        """Certificate with >825 days validity should be suspicious"""
        now = datetime.now()
        not_before = now.timestamp()
        not_after = (now + timedelta(days=900)).timestamp()
        
        cert_data = {
            'leaf_cert': {
                'not_before': not_before,
                'not_after': not_after
            }
        }
        self.assertEqual(f11_validity_period(cert_data), 1)
    
    def test_normal_validity(self):
        """Certificate with normal validity (90 days) should not be suspicious"""
        now = datetime.now()
        not_before = (now - timedelta(days=30)).timestamp()
        not_after = (now + timedelta(days=60)).timestamp()
        
        cert_data = {
            'leaf_cert': {
                'not_before': not_before,
                'not_after': not_after
            }
        }
        self.assertEqual(f11_validity_period(cert_data), 0)
    
    def test_recently_issued_brand_domain(self):
        """Recently issued cert for brand domain should be suspicious"""
        now = datetime.now()
        not_before = (now - timedelta(days=3)).timestamp()  # Issued 3 days ago
        not_after = (now + timedelta(days=87)).timestamp()
        
        cert_data = {
            'all_domains': ['paypal-login.com'],
            'leaf_cert': {
                'not_before': not_before,
                'not_after': not_after
            }
        }
        self.assertEqual(f11_validity_period(cert_data), 1)
    
    def test_recently_issued_normal_domain(self):
        """Recently issued cert for normal domain should not be suspicious"""
        now = datetime.now()
        not_before = (now - timedelta(days=3)).timestamp()
        not_after = (now + timedelta(days=87)).timestamp()
        
        cert_data = {
            'all_domains': ['mywebsite.com'],
            'leaf_cert': {
                'not_before': not_before,
                'not_after': not_after
            }
        }
        self.assertEqual(f11_validity_period(cert_data), 0)
    
    def test_missing_validity_data(self):
        """Missing validity data should return 0 (conservative)"""
        cert_data = {
            'leaf_cert': {}
        }
        self.assertEqual(f11_validity_period(cert_data), 0)

class TestChainValidation(unittest.TestCase):
    """Test F12: Certificate Chain Validation"""
    
    def test_valid_chain(self):
        """Valid certificate chain should not be suspicious"""
        cert_data = {
            'leaf_cert': {
                'subject': {'aggregated': '/CN=example.com'},
                'issuer': {'aggregated': '/CN=Intermediate CA'}
            },
            'data': {
                'chain': [
                    {'subject': {'aggregated': '/CN=Intermediate CA'},
                     'issuer': {'aggregated': '/CN=Root CA'}},
                    {'subject': {'aggregated': '/CN=Root CA'},
                     'issuer': {'aggregated': '/CN=Root CA'}}  # Self-signed root
                ]
            }
        }
        self.assertEqual(f12_chain_validation(cert_data), 0)
    
    def test_broken_chain(self):
        """Broken chain (issuer/subject mismatch) should be suspicious"""
        cert_data = {
            'leaf_cert': {
                'subject': {'aggregated': '/CN=example.com'},
                'issuer': {'aggregated': '/CN=Intermediate CA'}
            },
            'data': {
                'chain': [
                    {'subject': {'aggregated': '/CN=Wrong CA'},  # Mismatch!
                     'issuer': {'aggregated': '/CN=Root CA'}}
                ]
            }
        }
        self.assertEqual(f12_chain_validation(cert_data), 1)
    
    def test_short_chain_non_self_signed(self):
        """Short chain for non-self-signed cert should be suspicious"""
        cert_data = {
            'leaf_cert': {
                'subject': {'aggregated': '/CN=example.com'},
                'issuer': {'aggregated': '/CN=Some CA'}  # Not self-signed
            },
            'data': {
                'chain': []  # Empty chain
            }
        }
        self.assertEqual(f12_chain_validation(cert_data), 1)
    
    def test_self_signed_single_cert(self):
        """Self-signed single cert should not trigger F12 (caught by F10)"""
        cert_data = {
            'leaf_cert': {
                'subject': {'aggregated': '/CN=example.com'},
                'issuer': {'aggregated': '/CN=example.com'}  # Self-signed
            },
            'data': {
                'chain': []
            }
        }
        self.assertEqual(f12_chain_validation(cert_data), 0)
    
    def test_no_chain_data(self):
        """Missing chain data for non-self-signed cert should be suspicious"""
        cert_data = {
            'leaf_cert': {
                'subject': {'aggregated': '/CN=example.com'},
                'issuer': {'aggregated': '/CN=Some CA'}
            }
        }
        # Non-self-signed cert with no chain = suspicious
        self.assertEqual(f12_chain_validation(cert_data), 1)
    
    def test_chain_continuity_broken(self):
        """Chain with broken continuity should be suspicious"""
        cert_data = {
            'leaf_cert': {
                'subject': {'aggregated': '/CN=example.com'},
                'issuer': {'aggregated': '/CN=Intermediate CA'}
            },
            'data': {
                'chain': [
                    {'subject': {'aggregated': '/CN=Intermediate CA'},
                     'issuer': {'aggregated': '/CN=Root CA'}},
                    {'subject': {'aggregated': '/CN=Different CA'},  # Broken!
                     'issuer': {'aggregated': '/CN=Another Root'}}
                ]
            }
        }
        self.assertEqual(f12_chain_validation(cert_data), 1)



class TestCertSecurityFeatures(unittest.TestCase):
    """Test combined certificate security features"""
    
    def test_extract_all_features(self):
        """Test extracting all F9-F12 features"""
        now = datetime.now()
        cert_data = {
            'all_domains': ['example.com', 'www.example.com'],
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
            },
            'data': {
                'chain': [
                    {'subject': {'aggregated': '/C=US/O=Let\'s Encrypt/CN=R3'},
                     'issuer': {'aggregated': '/C=US/O=Internet Security Research Group/CN=ISRG Root X1'}},
                    {'subject': {'aggregated': '/C=US/O=Internet Security Research Group/CN=ISRG Root X1'},
                     'issuer': {'aggregated': '/C=US/O=Internet Security Research Group/CN=ISRG Root X1'}}
                ]
            }
        }
        
        features = extract_cert_security_features(cert_data)
        self.assertEqual(len(features), 4)
        self.assertEqual(features, [0, 0, 0, 0])  # All should be 0 (not suspicious)
    
    def test_suspicious_certificate(self):
        """Test suspicious certificate with multiple flags"""
        now = datetime.now()
        cert_data = {
            'all_domains': ['paypal-login.com', 'amazon-verify.com', 'bank-secure.com'],
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
            },
            'data': {'chain': []}
        }
        
        features = extract_cert_security_features(cert_data)
        self.assertEqual(len(features), 4)
        # F9: Multiple brands = 1
        # F10: Self-signed = 1
        # F11: Short validity + recently issued brand domain = 1
        # F12: Self-signed so chain OK = 0
        self.assertEqual(features, [1, 1, 1, 0])
    
    def test_analyze_certificate_security(self):
        """Test comprehensive certificate security analysis"""
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
        
        analysis = analyze_certificate_security(cert_data)
        
        self.assertIn('san_domains', analysis)
        self.assertIn('san_count', analysis)
        self.assertIn('f9_san_suspicious', analysis)
        self.assertIn('f10_self_signed', analysis)
        self.assertIn('f11_validity_suspicious', analysis)
        self.assertIn('validity_days', analysis)
        self.assertIn('certificate_age_days', analysis)
        self.assertIn('features', analysis)
        
        self.assertEqual(analysis['san_count'], 1)
        self.assertFalse(analysis['f9_san_suspicious'])
        self.assertFalse(analysis['f10_self_signed'])
        self.assertFalse(analysis['f11_validity_suspicious'])
        self.assertEqual(analysis['validity_days'], 90)
        self.assertEqual(analysis['certificate_age_days'], 30)


if __name__ == '__main__':
    unittest.main()
