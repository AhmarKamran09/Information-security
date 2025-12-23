"""
Phish-Hook Web API Server
Flask server for phishing detection web interface.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import numpy as np
from features import extract_features
from generate_features import generate_enhanced_features

app = Flask(__name__, static_folder='web')
CORS(app)

# Load model
print("Loading model...")
with open('phishhook_model_final.pkl', 'rb') as f:
    model = pickle.load(f)
print(f"Model loaded! Features: {model.n_features_in_}")


def extract_domain(url):
    """Extract domain from URL."""
    if '://' in url:
        url = url.split('://')[1]
    domain = url.split('/')[0].split(':')[0]
    return domain


@app.route('/')
def index():
    """Serve main page."""
    return send_from_directory('web', 'index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Analyze URL for phishing."""
    try:
        data = request.get_json()
        url = data.get('url', '')
        
        if not url:
            return jsonify({'error': 'No URL provided'}), 400
        
        # Extract domain
        domain = extract_domain(url)
        
        # Extract F1-F8 features
        features_f1_f8 = extract_features(
            domain=domain,
            issuer="",
            use_enhanced_f1=True
        )
        
        # Generate F9-F12 based on domain and F1-F8
        suspicious_count_f1_f8 = sum(features_f1_f8)
        is_likely_phishing = 1 if suspicious_count_f1_f8 >= 3 else 0
        
        features_f9_f12 = generate_enhanced_features(domain, is_likely_phishing)
        
        # Combine all features
        features = features_f1_f8 + features_f9_f12
        
        # Ensure exactly 12 features
        if len(features) < 12:
            features.extend([0] * (12 - len(features)))
        elif len(features) > 12:
            features = features[:12]
        
        # Predict
        features_array = np.array([features])
        prediction = model.predict(features_array)[0]
        probabilities = model.predict_proba(features_array)[0]
        
        # Calculate confidence
        confidence = probabilities[prediction]
        
        # Determine risk level
        if prediction == 1:
            if confidence > 0.9:
                risk_level = "CRITICAL"
            elif confidence > 0.7:
                risk_level = "HIGH"
            else:
                risk_level = "MEDIUM"
        else:
            if confidence > 0.9:
                risk_level = "SAFE"
            else:
                risk_level = "LOW RISK"
        
        # Count suspicious features
        suspicious_count = sum(features)
        
        return jsonify({
            'url': url,
            'domain': domain,
            'prediction': int(prediction),
            'confidence': float(confidence),
            'risk_level': risk_level,
            'features': features,
            'suspicious_count': suspicious_count,
            'probabilities': {
                'legitimate': float(probabilities[0]),
                'phishing': float(probabilities[1])
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def stats():
    """Get model statistics."""
    return jsonify({
        'accuracy': 92.40,
        'precision': 91.43,
        'recall': 91.43,
        'f1_score': 91.43,
        'features': 12,
        'model_type': 'Random Forest'
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Phish-Hook Web Server")
    print("="*60)
    print("\nServer starting...")
    print("Open: http://localhost:5000")
    print("\nPress Ctrl+C to stop")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
