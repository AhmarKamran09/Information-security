
# Phish-Hook: Real-Time Phishing Detection System

Phish-Hook is an advanced, real-time phishing detection system that monitors Certificate Transparency (CT) logs to identify potential phishing domains as they are issued. Leveraging machine learning and branding similarity analysis, it classifies domains into risk levels to protect users from emerging threats.

## 🚀 Features

-   **Real-Time Monitoring**: Listens to CertStream for live SSL/TLS certificate issuance events.
-   **Machine Learning Classification**: Uses a trained **Support Vector Machine (SVM)** (or similar high-performance classifiers) to evaluate domains based on **12 distinct features**.
    -   **F1-F8**: Structural and lexical features (e.g., URL length, suspicious patterns, hyphens).
    -   **F9-F12**: Certificate security features and metadata.
-   **Brand Similarity Detection**: Enhanced "F1" feature uses vector-based embedding and Levenshtein distance to detect typosquatting and visual look-alikes of popular brands.
-   **Campaign Detection**: Automatically clusters related phishing domains into "campaigns" based on time windows and similarity heuristics, catching large-scale attacks.
-   **Risk Scoring**: Classifies domains into 5 granular risk levels:
    1.  **Legitimate** (0-20%)
    2.  **Potential** (20-40%)
    3.  **Likely** (40-60%)
    4.  **Suspicious** (60-80%)
    5.  **Highly Suspicious** (80-100%)

## 📊 Model Metrics

The core detection model (`phishhook_model_final.pkl`) achieves high performance on the UCI Phishing Dataset (augmented with certificate features):

| Metric | Score | Matches |
| :--- | :--- | :--- |
| **Accuracy** | **88.01%** | Overall correctness of predictions |
| **Precision** | **94.63%** | High confidence in flagged phishing sites (Low False Positives) |
| **Recall** | **77.35%** | Ability to detect actual phishing attempts |
| **F1 Score** | **85.12%** | Balanced performance metric |

*> Note: Metrics are based on the latest evaluation on the test set (20% split).*

## 🛠️ Usage

### Prerequisites
Install dependencies:
```bash
pip install -r requirements.txt
```

### 1. Start Real-Time Detection
Run the server to start monitoring CertStream:
```bash
python3 serve.py --enable-campaign
```
Options:
-   `--enable-campaign`: Turn on campaign clustering.
-   `--output <file>`: Save detections to a JSON line file.
-   `--domain <domain>`: Test a specific domain string manually.

### 2. Train/Retrain Model
To train the model with improved features:
```bash
python3 train_improved.py --dataset uci-ml-phishing-dataset.csv --plot-curves
```

### 3. Evaluate Model
Generate a full performance report:
```bash
python3 evaluation.py --model phishhook_model_final.pkl
```

## 📂 Project Structure

-   `serve.py`: Main entry point for real-time detection and server logic.
-   `train_improved.py`: Training script with enhanced feature engineering (brand embeddings).
-   `evaluation.py`: Model evaluation and reporting tools.
-   `campaign.py`: Logic for detecting and clustering phishing campaigns.
-   `features.py` & `cert_security.py`: Feature extraction modules.
-   `uci_mapper.py`: Helper to map UCI dataset features to Phish-Hook format.
