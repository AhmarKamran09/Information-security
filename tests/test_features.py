"""
Quick test script to verify feature extraction works correctly.
"""

from features import extract_features

# Test domains
test_domains = [
    ("google.com", ""),  # Legitimate
    ("googla.com", ""),  # Look-alike (F1)
    ("secure.login.verify.account.example.com", ""),  # Deep subdomains (F2)
    ("example.ga", "Let's Encrypt"),  # Suspicious TLD + Free CA (F3, F4)
    ("paypal-com-login.example.com", ""),  # Inner TLD (F5)
    ("login-verify-bank.example.com", ""),  # Suspicious keywords (F6)
    ("randomx92jsk3.example.com", ""),  # High entropy (F7)
    ("paypal-secure-login.example.com", ""),  # Hyphens (F8)
]

print("="*60)
print("Feature Extraction Test")
print("="*60)

for domain, issuer in test_domains:
    features = extract_features(domain, issuer)
    print(f"\nDomain: {domain}")
    print(f"Issuer: {issuer}")
    print(f"Features: F1={features[0]} F2={features[1]} F3={features[2]} "
          f"F4={features[3]} F5={features[4]} F6={features[5]} "
          f"F7={features[6]} F8={features[7]}")
    print(f"Feature vector: {features}")

print("\n" + "="*60)
print("Test Complete!")
print("="*60)

