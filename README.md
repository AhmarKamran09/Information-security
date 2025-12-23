# Phish-Hook - Advanced Phishing Detection System

[![Accuracy](https://img.shields.io/badge/Accuracy-92.18%25-brightgreen)](models/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

Advanced machine learning-based phishing detection system with 92.18% accuracy using domain analysis and certificate security features.

## 📊 Performance

- **Accuracy:** 92.18%
- **Precision:** 91.38%
- **Recall:** 90.92%
- **F1-Score:** 91.15%
- **ROC-AUC:** 96.86%

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/AhmarKamran09/Information-security.git
cd Information-security

# Install dependencies
pip install -r requirements.txt
```

### Usage

**Evaluate Model:**
```bash
python scripts/evaluate.py
```

**Train Model:**
```bash
python scripts/train.py
```

**Run Web Interface:**
```bash
python web/web_server.py
```
Then open: http://localhost:5000

## 📁 Project Structure

```
phish-hook/
├── models/              # Trained models
│   └── phishhook_model.pkl
├── data/                # Training datasets
│   └── training_data.npz
├── scripts/             # Training & evaluation scripts
│   ├── train.py
│   └── evaluate.py
├── web/                 # Web interface
│   ├── web_server.py
│   └── index.html
├── tests/               # Unit tests
├── docs/                # Documentation
├── features.py          # Feature extraction
├── cert_security.py     # Certificate security features
├── uci_mapper.py        # Dataset utilities
└── requirements.txt     # Dependencies
```

## 🎯 Features

### Domain Features (F1-F8)
- F1: Brand Similarity Detection
- F2: Subdomain Depth Analysis
- F3: Free Certificate Authority Detection
- F4: Suspicious TLD Detection
- F5: Inner TLD in Subdomain
- F6: Suspicious Keywords
- F7: High Entropy Detection
- F8: Hyphens in Subdomain

### Certificate Security Features (F9-F12)
- F9: SAN (Subject Alternative Names) Analysis
- F10: Self-Signed Certificate Detection
- F11: Validity Period Analysis
- F12: Certificate Chain Validation

## 🔬 Model Details

- **Algorithm:** Random Forest Classifier
- **Trees:** 200
- **Features:** 12 (F1-F12)
- **Training Samples:** 11,055
- **Improvement:** +4.66% over baseline

## 📈 Results

```
Confusion Matrix:
                 Predicted
                 Legit  Phishing
  Actual Legit    1147      84
  Actual Phish      89     891

True Positives:  891 (correctly identified phishing)
True Negatives:  1147 (correctly identified legitimate)
False Positives: 84 (legitimate flagged as phishing)
False Negatives: 89 (phishing missed)
```

## 🌐 Web Interface

Beautiful, modern web interface for real-time phishing detection:
- Instant URL analysis
- Visual feature breakdown
- Risk level assessment
- Confidence scoring

## 📝 License

MIT License - see LICENSE file for details

## 👥 Authors

- Ahmar Kamran

## 🙏 Acknowledgments

Based on research from TU Graz phishing detection paper.
