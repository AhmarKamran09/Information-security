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
from campaign import CampaignDetector


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
            model_data = pickle.load(f)
        
        # Handle both formats: dict (advanced) or direct model (standard)
        if isinstance(model_data, dict) and 'model' in model_data:
            self.model = model_data['model']
            self.scaler = model_data.get('scaler', None)
            print(f"Model loaded successfully! (Type: {model_data.get('model_name', 'Unknown')})")
        else:
            self.model = model_data
            self.scaler = None
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
        
        # Apply scaler if available (from advanced training)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
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
    
    def classify_domain(self, domain: str, issuer: str = "", use_enhanced_f1: bool = True) -> Dict:
        """
        Classify a single domain.
        
        Args:
            domain: Domain string
            issuer: Certificate issuer
            use_enhanced_f1: Whether to use enhanced brand similarity (embedding-based)
            
        Returns:
            Classification results
        """
        features = extract_features(domain, issuer, use_enhanced_f1=use_enhanced_f1)
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
            return {}
        
        self.stats['total_domains'] += len(domains)
        
        # Classify each domain and collect classification mapping
        classification: Dict[str, Dict] = {}
        for i, domain in enumerate(domains):
            if i < len(features_list):
                features = features_list[i]
            else:
                # Fallback: extract features if not provided
                features = extract_features(domain, issuer)

            result = self.predict_risk_level(features)
            result['domain'] = domain
            result['issuer'] = issuer

            classification[domain] = result

            # Update statistics
            self.stats['risk_levels'][result['risk_level']] += 1
            if result['prediction'] == 1 or result['phishing_probability'] >= 0.5:
                self.stats['phishing_detected'] += 1
            else:
                self.stats['legitimate'] += 1

            # Print high-risk domains (for operator situational awareness)
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

        # Attach classification mapping for downstream consumers
        cert_data['classification'] = classification
        return classification
    
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
        # Campaign detector is disabled by default; can be enabled by caller
        self.campaign_detector: Optional[CampaignDetector] = None
    
    def callback(self, cert_data: Dict):
        """Callback for certificate processing."""
        classification = self.detector.process_certificate(cert_data)

        # Save to file if specified (include classification)
        if self.output_file:
            with open(self.output_file, 'a') as f:
                f.write(json.dumps({
                    'timestamp': cert_data.get('timestamp'),
                    'domains': cert_data.get('domains'),
                    'issuer': cert_data.get('issuer'),
                    'classification': classification
                }) + '\n')

        # Campaign detection (if configured)
        if self.campaign_detector is not None:
            alerts = self.campaign_detector.add(cert_data, classification)
            for alert in alerts:
                print('\n' + '!'*10 + ' CAMPAIGN ALERT ' + '!'*10)
                print(f"Campaign key: {alert.get('campaign_key')}")
                print(f"Size: {alert.get('size')}  Unique domains: {alert.get('unique_domains')}")
                print(f"Phishing count: {alert.get('phishing_count')}, Avg prob: {alert.get('average_phishing_probability')}")
                print('!'*36 + '\n')

        # Print stats periodically
        if self.detector.stats['total_domains'] % 10 == 0:
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
    parser.add_argument('--no-enhanced-f1', action='store_true',
                       help='Disable enhanced brand similarity (use original Levenshtein)')
    parser.add_argument('--enable-campaign', action='store_true',
                       help='Enable campaign-level detection (grouping heuristics)')
    parser.add_argument('--campaign-threshold', type=int, default=3,
                       help='Minimum campaign size to trigger alert')
    parser.add_argument('--campaign-window', type=int, default=86400,
                       help='Time window for campaign clustering (seconds)')
    
    args = parser.parse_args()
    
    detector = PhishHookDetector(args.model)
    
    # Test single domain if provided
    if args.domain:
        print(f"\nTesting domain: {args.domain}")
        use_enhanced = not args.no_enhanced_f1
        if use_enhanced:
            print("Using enhanced F1 (brand similarity embedding)")
        result = detector.classify_domain(args.domain, use_enhanced_f1=use_enhanced)
        print(f"\nResult:")
        print(f"  Domain: {result['domain']}")
        print(f"  Risk Level: {result['risk_level']} - {result['risk_name']}")
        print(f"  Phishing Probability: {result['phishing_probability']:.2%}")
        print(f"  Features: {result['features']}")
        return
    
    # Start real-time detection
    rt_detector = RealTimeDetector(args.model, args.output)
    if args.enable_campaign:
        print(f"Enabling campaign detection: threshold={args.campaign_threshold}, window={args.campaign_window}s")
        rt_detector.campaign_detector = CampaignDetector(size_threshold=args.campaign_threshold,
                                                         time_window_seconds=args.campaign_window)
    rt_detector.start()


if __name__ == '__main__':
    main()

