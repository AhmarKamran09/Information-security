# Phish-Hook: Phishing Detection at Certificate Issuance

A complete implementation of the Phish-Hook paper for detecting phishing websites at the time of certificate issuance using Certificate Transparency logs and traditional machine learning.

## Overview

Phish-Hook detects phishing websites by:
- Extracting 8 lexical and certificate features (F1-F8) from Certificate Transparency logs
- Training traditional ML models (SVM, Decision Tree, KNN, MLP) on the UCI Phishing Dataset
- Classifying domains in real-time with 5-level risk scoring

**Key Features:**
- ✅ Real-time CT log monitoring via CertStream
- ✅ 8 feature extraction (F1-F8)
- ✅ SMOTE + Random Undersampling for class balancing
- ✅ SVM with optimal hyperparameters (C=0.03, linear kernel)
- ✅ 5-level risk scoring (Legitimate → Highly Suspicious)
- ✅ Expected performance: ~95% accuracy, ~94-95% F1 score

## Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── features.py              # F1-F8 feature extraction
├── uci_mapper.py            # UCI dataset to F1-F8 mapper
├── train.py                 # Model training script
├── collector.py             # CertStream CT log collector
├── serve.py                 # Real-time detection server
├── evaluation.py            # Model evaluation script
├── uci-ml-phishing-dataset.csv  # Training dataset
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Usage

### 1. Train the Model

Train models on the UCI dataset:

```bash
python train.py --dataset uci-ml-phishing-dataset.csv --model-output phishhook_model.pkl
```

**Options:**
- `--dataset`: Path to UCI phishing dataset CSV (default: `uci-ml-phishing-dataset.csv`)
- `--model-output`: Output path for trained model (default: `phishhook_model.pkl`)
- `--cv-folds`: Number of cross-validation folds (default: 10)
- `--test-size`: Test set size 0.0-1.0 (default: 0.2)

**Expected Output:**
- Trained model saved to `phishhook_model.pkl`
- Performance metrics matching paper results (~95% accuracy)

### 2. Evaluate the Model

Evaluate trained model performance:

```bash
python evaluation.py --model phishhook_model.pkl --dataset uci-ml-phishing-dataset.csv
```

**Options:**
- `--model`: Path to trained model (default: `phishhook_model.pkl`)
- `--dataset`: Path to UCI dataset (default: `uci-ml-phishing-dataset.csv`)
- `--plot-roc`: Plot ROC curve
- `--roc-output`: ROC curve output file (default: `roc_curve.png`)

### 3. Real-Time Detection

Monitor Certificate Transparency logs in real-time:

```bash
python serve.py --model phishhook_model.pkl
```

**Options:**
- `--model`: Path to trained model (default: `phishhook_model.pkl`)
- `--output`: Optional file to save detections
- `--domain`: Test single domain (for testing)

**Example:**
```bash
# Test single domain
python serve.py --model phishhook_model.pkl --domain "paypal-secure-login.example.com"

# Real-time monitoring
python serve.py --model phishhook_model.pkl --output detections.jsonl
```

### 4. Collect CT Logs (Optional)

Collect CT logs without classification:

```bash
python collector.py --output ct_logs.jsonl --verbose
```

**Options:**
- `--output`: Output file for certificates (JSON lines format)
- `--verbose`: Print each certificate

## Feature Extraction (F1-F8)

The system extracts 8 binary features from domains:

| Feature | Name | Description |
|---------|------|-------------|
| F1 | Small Levenshtein Distance | Look-alike domain detection |
| F2 | Deeply Nested Subdomains | Multiple subdomain abuse |
| F3 | Issued from Free CA | Let's Encrypt, cPanel, Cloudflare, ZeroSSL |
| F4 | Suspicious TLD | .ga, .gdn, .xyz, .top, .win, etc. |
| F5 | Inner TLD in Subdomain | Fake "com", "org", "net" in subdomain |
| F6 | Suspicious Keywords | login, verify, update, secure, account, etc. |
| F7 | High Shannon Entropy | Random algorithmic domain |
| F8 | Hyphens in Subdomain | Phishing-style names (≥2 hyphens) |

## Risk Levels

The system classifies domains into 5 risk levels:

| Level | Name | Probability Range |
|-------|------|-------------------|
| 0 | Legitimate | 0.0 - 0.2 |
| 1 | Potential | 0.2 - 0.4 |
| 2 | Likely | 0.4 - 0.6 |
| 3 | Suspicious | 0.6 - 0.8 |
| 4 | Highly Suspicious | 0.8 - 1.0 |

## Model Performance

**Expected Performance (from paper):**
- Accuracy: ~95%
- Precision: ~93-96%
- Recall: ~93-95%
- F1 Score: ~94-95%

**Best Model:**
- SVM with linear kernel, C=0.03

## Training Pipeline

1. **Load UCI Dataset** (11,055 samples, 30 features)
2. **Map to F1-F8 Features** (convert UCI features to Phish-Hook features)
3. **SMOTE Oversampling** (oversample phishing class)
4. **Random Undersampling** (undersample legitimate class)
5. **Train Models** (SVM, Decision Tree, KNN, MLP)
6. **10-Fold Cross-Validation**
7. **Select Best Model** (highest F1 score)

## Real-Time Detection Pipeline

1. **CertStream** receives new certificate
2. **Extract Domains** from SAN (Subject Alternative Names)
3. **Compute F1-F8** features for each domain
4. **Feed into SVM** model
5. **Convert to Risk Level** (0-4) using probability thresholds
6. **Alert** on high-risk domains (Level 2+)

## Example Output

```
================================================================
⚠️  SUSPICIOUS DOMAIN DETECTED
================================================================
Domain: paypal-secure-login.example.ga
Issuer: Let's Encrypt
Risk Level: 4 - Highly Suspicious
Phishing Probability: 92.34%
Features: F1=1 F2=1 F3=1 F4=1 F5=1 F6=1 F7=0 F8=1
================================================================
```

## Technical Details

### Data Sources
- **Training**: UCI Phishing Dataset (11,055 samples)
- **Real-time**: Certificate Transparency logs via CertStream API
  - WebSocket: `wss://certstream.calidog.io/`

### Class Balancing
- **SMOTE**: Oversamples phishing class
- **Random Undersampling**: Undersamples legitimate class
- Both methods are required for matching paper results

### Models Supported
- SVM (Support Vector Machine) - **Primary/Best**
- Decision Tree
- K-Nearest Neighbors (KNN)
- Multi-Layer Perceptron (MLP)

## Limitations

- Does NOT crawl websites or analyze HTML
- Does NOT use deep learning or BERT
- Does NOT perform DNS resolution or IP reputation checks
- Does NOT use WHOIS data
- Purely lexical and certificate-based features

## Citation

If you use this implementation, please cite the original Phish-Hook paper.

## License

This implementation is provided for research and educational purposes.

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'certstream'`
- **Solution**: Install dependencies: `pip install -r requirements.txt`

**Issue**: Model performance below expected (~95%)
- **Solution**: Ensure SMOTE and undersampling are both applied during training

**Issue**: CertStream connection errors
- **Solution**: Check internet connection and CertStream API availability

## Contact

For questions or issues, please refer to the original paper or create an issue in the repository.

