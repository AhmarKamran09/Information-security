"""
Unit tests for campaign-level detection
"""
from campaign import CampaignDetector
from datetime import datetime, timedelta


def make_cert(domains, ips_map=None, ts=None):
    return {
        'timestamp': (ts or datetime.utcnow()).isoformat(),
        'domains': domains,
        'ips': ips_map or {},
    }


def make_classification(domains, prob=0.9):
    return {d: {'phishing_probability': prob, 'prediction': 1} for d in domains}


def test_campaign_by_ip():
    cd = CampaignDetector(size_threshold=2, time_window_seconds=3600)

    c1 = make_cert(['phishy1.example.com'], ips_map={'phishy1.example.com': ['1.2.3.4']})
    c2 = make_cert(['login-secure.example.com'], ips_map={'login-secure.example.com': ['1.2.3.4']})

    alerts1 = cd.add(c1, make_classification(['phishy1.example.com']))
    assert alerts1 == []

    alerts2 = cd.add(c2, make_classification(['login-secure.example.com']))
    # Should detect campaign by shared IP
    assert len(alerts2) >= 1
    a = alerts2[0]
    assert a['size'] >= 2
    assert 'phishy1.example.com' in a['unique_domains']


def test_campaign_by_shortener():
    cd = CampaignDetector(size_threshold=2, time_window_seconds=3600)

    # Use a shortener key by domain host
    c1 = make_cert(['bit.ly'], ips_map={'bit.ly': ['8.8.8.8']})
    c2 = make_cert(['bit.ly/abcd'], ips_map={'bit.ly/abcd': ['8.8.8.8']})

    alerts1 = cd.add(c1, make_classification(['bit.ly']))
    assert alerts1 == []

    alerts2 = cd.add(c2, make_classification(['bit.ly/abcd']))
    assert len(alerts2) >= 1
