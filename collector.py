"""
Phish-Hook CertStream Collector
Collects Certificate Transparency logs in real-time via CertStream API.
"""

import certstream
import json
import time
from datetime import datetime
from typing import Callable, Optional
from features import extract_features_from_ct_entry


class CTLogCollector:
    """Collects CT logs from CertStream."""
    
    def __init__(self, callback: Optional[Callable] = None, output_file: Optional[str] = None):
        """
        Initialize CT Log Collector.
        
        Args:
            callback: Optional callback function to process each certificate
            output_file: Optional file path to save certificates
        """
        self.callback = callback
        self.output_file = output_file
        self.cert_count = 0
        self.start_time = None
        
    def process_certificate(self, message, context):
        """Process a certificate update message."""
        if message['message_type'] != 'certificate_update':
            return
        
        try:
            data = message['data']
            leaf_cert = data.get('leaf_cert', {})
            domains = leaf_cert.get('all_domains', [])
            
            if not domains:
                return
            
            # Extract issuer
            chain = data.get('chain', [])
            issuer = ""
            if chain and len(chain) > 0:
                issuer = chain[0].get('subject', {}).get('aggregated', '')
            
            # Extract features
            features_list, domains_list = extract_features_from_ct_entry(message)
            
            cert_data = {
                'timestamp': datetime.now().isoformat(),
                'domains': domains_list,
                'issuer': issuer,
                'features': features_list,
                'cert_index': data.get('cert_index', None)
            }
            
            self.cert_count += 1
            
            # Save to file if specified
            if self.output_file:
                with open(self.output_file, 'a') as f:
                    f.write(json.dumps(cert_data) + '\n')
            
            # Call callback if provided
            if self.callback:
                self.callback(cert_data)
            
            # Print progress every 100 certificates
            if self.cert_count % 100 == 0:
                elapsed = time.time() - self.start_time if self.start_time else 0
                rate = self.cert_count / elapsed if elapsed > 0 else 0
                print(f"[{datetime.now()}] Collected {self.cert_count} certificates "
                      f"({rate:.2f} certs/sec)")
        
        except Exception as e:
            print(f"Error processing certificate: {e}")
    
    def start(self):
        """Start collecting certificates."""
        print("="*60)
        print("Phish-Hook CertStream Collector")
        print("="*60)
        print("\nConnecting to CertStream API...")
        print("WebSocket: wss://certstream.calidog.io/")
        print("\nCollecting certificates... (Press Ctrl+C to stop)\n")
        
        self.start_time = time.time()
        
        try:
            certstream.listen_for_events(
                self.process_certificate,
                url='wss://certstream.calidog.io/'
            )
        except KeyboardInterrupt:
            print("\n\nStopping collector...")
            elapsed = time.time() - self.start_time
            print(f"\nCollection Summary:")
            print(f"  Total certificates: {self.cert_count}")
            print(f"  Time elapsed: {elapsed:.2f} seconds")
            print(f"  Average rate: {self.cert_count/elapsed:.2f} certs/sec")
            if self.output_file:
                print(f"  Saved to: {self.output_file}")


def simple_callback(cert_data):
    """Simple callback to print certificate info."""
    domains = cert_data['domains']
    if domains:
        print(f"Domain: {domains[0]} | Issuer: {cert_data['issuer']}")


def main():
    """Main function for standalone collector."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect CT logs via CertStream')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file to save certificates (JSON lines)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print each certificate')
    
    args = parser.parse_args()
    
    callback = simple_callback if args.verbose else None
    collector = CTLogCollector(callback=callback, output_file=args.output)
    collector.start()


if __name__ == '__main__':
    main()

