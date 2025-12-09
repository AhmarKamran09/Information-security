"""
Phish-Hook Real-Time Detection Server
Monitors Certificate Transparency logs and classifies domains in real-time.
"""

import pickle
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
from collector import CTLogCollector
from features import extract_features_from_ct_entry, extract_features


class PhishHookDetector:
    """Real-time phishing detection using trained model."""
    
    # Risk level thresholds (based on SVM probability)
    RISK_THRESHOLDS = {
        0: (0.0, 0.2),      # Legitimate
        1: (0.2, 0.4),      # Potential
        2: (0.4, 0.6),      # Likely
        3: (0.6, 0.8),      # Suspicious
        4: (0.8, 1.0)       # Highly Suspicious
    }
    
    RISK_NAMES = {
        0: "Legitimate",
        1: "Potential",
        2: "Likely",
        3: "Suspicious",
        4: "Highly Suspicious"
    }
    
    def __init__(self, model_path: str):
        """
        Initialize detector with trained model.
        
        Args:
            model_path: Path to saved model pickle file
        """
        print(f"Loading model from {model_path}...")
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        print("Model loaded successfully!")
        
        self.stats = {
            'total_domains': 0,
            'risk_levels': {i: 0 for i in range(5)},
            'phishing_detected': 0,
            'legitimate': 0
        }
    
    def predict_risk_level(self, features: List[int]) -> Dict:
        """
        Predict risk level for given features.
        
        Args:
            features: List of F1-F8 features
            
        Returns:
            Dictionary with prediction results
        """
        if len(features) != 8:
            raise ValueError(f"Expected 8 features, got {len(features)}")
        
        # Reshape for prediction
        X = np.array(features).reshape(1, -1)
        
        # Get prediction and probability
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        # Get phishing probability (class 1)
        phishing_prob = probabilities[1] if len(probabilities) > 1 else probabilities[0]
        
        # Determine risk level
        risk_level = 0
        for level, (low, high) in self.RISK_THRESHOLDS.items():
            if low <= phishing_prob < high:
                risk_level = level
                break
        if phishing_prob >= 0.8:
            risk_level = 4
        
        return {
            'prediction': int(prediction),
            'phishing_probability': float(phishing_prob),
            'risk_level': risk_level,
            'risk_name': self.RISK_NAMES[risk_level],
            'features': features
        }
    
    def classify_domain(self, domain: str, issuer: str = "") -> Dict:
        """
        Classify a single domain.
        
        Args:
            domain: Domain string
            issuer: Certificate issuer
            
        Returns:
            Classification results
        """
        features = extract_features(domain, issuer)
        result = self.predict_risk_level(features)
        result['domain'] = domain
        result['issuer'] = issuer
        
        return result
    
    def process_certificate(self, cert_data: Dict):
        """Process a certificate and classify domains."""
        domains = cert_data.get('domains', [])
        issuer = cert_data.get('issuer', '')
        features_list = cert_data.get('features', [])
        
        if not domains:
            return
        
        self.stats['total_domains'] += len(domains)
        
        # Classify each domain
        for i, domain in enumerate(domains):
            if i < len(features_list):
                features = features_list[i]
            else:
                # Fallback: extract features if not provided
                features = extract_features(domain, issuer)
            
            result = self.predict_risk_level(features)
            
            # Update statistics
            self.stats['risk_levels'][result['risk_level']] += 1
            if result['prediction'] == 1:
                self.stats['phishing_detected'] += 1
            else:
                self.stats['legitimate'] += 1
            
            # Print high-risk domains
            if result['risk_level'] >= 2:  # Likely, Suspicious, or Highly Suspicious
                print(f"\n{'='*60}")
                print(f"⚠️  SUSPICIOUS DOMAIN DETECTED")
                print(f"{'='*60}")
                print(f"Domain: {domain}")
                print(f"Issuer: {issuer}")
                print(f"Risk Level: {result['risk_level']} - {result['risk_name']}")
                print(f"Phishing Probability: {result['phishing_probability']:.2%}")
                print(f"Features: F1={features[0]} F2={features[1]} F3={features[2]} "
                      f"F4={features[3]} F5={features[4]} F6={features[5]} "
                      f"F7={features[6]} F8={features[7]}")
                print(f"{'='*60}\n")
    
    def print_stats(self):
        """Print detection statistics."""
        print("\n" + "="*60)
        print("Detection Statistics")
        print("="*60)
        print(f"Total domains analyzed: {self.stats['total_domains']}")
        print(f"Phishing detected: {self.stats['phishing_detected']}")
        print(f"Legitimate: {self.stats['legitimate']}")
        print("\nRisk Level Distribution:")
        for level in range(5):
            count = self.stats['risk_levels'][level]
            percentage = (count / self.stats['total_domains'] * 100) if self.stats['total_domains'] > 0 else 0
            print(f"  Level {level} ({self.RISK_NAMES[level]}): {count} ({percentage:.2f}%)")


class RealTimeDetector:
    """Real-time detection using CertStream."""
    
    def __init__(self, model_path: str, output_file: Optional[str] = None):
        """
        Initialize real-time detector.
        
        Args:
            model_path: Path to trained model
            output_file: Optional file to save detections
        """
        self.detector = PhishHookDetector(model_path)
        self.output_file = output_file
        self.start_time = time.time()
    
    def callback(self, cert_data: Dict):
        """Callback for certificate processing."""
        self.detector.process_certificate(cert_data)
        
        # Save to file if specified
        if self.output_file:
            with open(self.output_file, 'a') as f:
                f.write(json.dumps(cert_data) + '\n')
        
        # Print stats every 1000 domains
        if self.detector.stats['total_domains'] % 1000 == 0:
            elapsed = time.time() - self.start_time
            rate = self.detector.stats['total_domains'] / elapsed if elapsed > 0 else 0
            print(f"\n[{datetime.now()}] Processed {self.detector.stats['total_domains']} domains "
                  f"({rate:.2f} domains/sec)")
    
    def start(self):
        """Start real-time detection."""
        print("="*60)
        print("Phish-Hook Real-Time Detection")
        print("="*60)
        print("\nStarting CertStream monitoring...")
        print("Press Ctrl+C to stop\n")
        
        collector = CTLogCollector(callback=self.callback, output_file=self.output_file)
        
        try:
            collector.start()
        except KeyboardInterrupt:
            print("\n\nStopping detection...")
            self.detector.print_stats()


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time phishing detection')
    parser.add_argument('--model', type=str, default='phishhook_model.pkl',
                       help='Path to trained model')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for detections')
    parser.add_argument('--domain', type=str, default=None,
                       help='Test single domain (for testing)')
    
    args = parser.parse_args()
    
    detector = PhishHookDetector(args.model)
    
    # Test single domain if provided
    if args.domain:
        print(f"\nTesting domain: {args.domain}")
        result = detector.classify_domain(args.domain)
        print(f"\nResult:")
        print(f"  Domain: {result['domain']}")
        print(f"  Risk Level: {result['risk_level']} - {result['risk_name']}")
        print(f"  Phishing Probability: {result['phishing_probability']:.2%}")
        print(f"  Features: {result['features']}")
        return
    
    # Start real-time detection
    rt_detector = RealTimeDetector(args.model, args.output)
    rt_detector.start()


if __name__ == '__main__':
    main()

