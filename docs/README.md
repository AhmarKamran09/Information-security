# Phish-Hook - Phishing Detection System

## Model Performance
- **Accuracy:** 92.40%
- **Precision:** 91.43%
- **Recall:** 91.43%
- **F1-Score:** 91.43%
- **ROC-AUC:** 97.09%

## Features (F1-F12)
**Domain Features (F1-F8):**
- F1: Brand Similarity
- F2: Subdomain Depth
- F3: Free CA Detection
- F4: Suspicious TLD
- F5: Inner TLD
- F6: Suspicious Keywords
- F7: High Entropy
- F8: Hyphens in Subdomain

**Certificate Security Features (F9-F12):**
- F9: SAN Analysis
- F10: Self-Signed Certificate
- F11: Validity Period
- F12: Chain Validation

## Quick Start

### 1. Train Model
```bash
python train.py
```

### 2. Evaluate Model
```bash
python evaluate.py
```

### 3. Run Web Interface
```bash
python web_server.py
```
Then open: http://localhost:5000

## Files
- `phishhook_model.pkl` - Trained model (92.40% accuracy)
- `train.py` - Training script
- `evaluate.py` - Evaluation script with full metrics
- `web_server.py` - Web interface server
- `features.py` - Feature extraction
- `cert_security.py` - Certificate security features

## Model Details
- **Algorithm:** Random Forest (200 trees)
- **Features:** 12 (F1-F12)
- **Dataset:** UCI Phishing Dataset (11,055 samples)
- **Improvement:** +4.88% over baseline
