"""
Tests for evaluation campaign analysis
"""
import json
from evaluation import analyze_campaigns_from_stream, print_campaign_report
from datetime import datetime
import tempfile


def run_smoke_test():
    now = datetime.utcnow().isoformat()
    entries = [
        {'timestamp': now, 'domains': ['a.example.com'], 'issuer': '', 'ips': {'a.example.com': ['1.2.3.4']}, 'classification': {'a.example.com': {'phishing_probability': 0.9, 'prediction': 1}}},
        {'timestamp': now, 'domains': ['b.example.com'], 'issuer': '', 'ips': {'b.example.com': ['1.2.3.4']}, 'classification': {'b.example.com': {'phishing_probability': 0.8, 'prediction': 1}}}
    ]
    with tempfile.NamedTemporaryFile('w+', delete=False) as tf:
        for e in entries:
            tf.write(json.dumps(e) + '\n')
        tf.flush()
        alerts = analyze_campaigns_from_stream(tf.name, size_threshold=2)
        assert len(alerts) >= 1
        print_campaign_report(alerts)


if __name__ == '__main__':
    run_smoke_test()
