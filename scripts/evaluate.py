"""
Phish-Hook Model Evaluation Script
Displays comprehensive metrics for the trained model.
"""

import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt


def print_section(title):
    """Print a section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def evaluate_model():
    """Evaluate the trained model and display metrics."""
    
    print("\n" + "="*70)
    print("  PHISH-HOOK MODEL EVALUATION")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    data = np.load('../data/training_data.npz')
    X = data['X']
    y = data['y']
    
    print(f"  Dataset: {len(X)} samples")
    print(f"  Features: {X.shape[1]} (F1-F12)")
    print(f"  Phishing: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
    print(f"  Legitimate: {len(y)-sum(y)} ({(len(y)-sum(y))/len(y)*100:.1f}%)")
    
    # Load model
    print("\nLoading model...")
    with open('../models/phishhook_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    print(f"  Model type: {type(model).__name__}")
    print(f"  Expected features: {model.n_features_in_}")
    
    # Split data (same as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTest set: {len(X_test)} samples")
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    print_section("PERFORMANCE METRICS")
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
    except:
        auc = 0.0
    
    print(f"\n  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"  ROC-AUC:   {auc:.4f} ({auc*100:.2f}%)")
    
    # Confusion Matrix
    print_section("CONFUSION MATRIX")
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n                 Predicted")
    print(f"                 Legit  Phishing")
    print(f"  Actual Legit    {tn:4d}    {fp:4d}")
    print(f"  Actual Phish    {fn:4d}    {tp:4d}")
    
    print(f"\n  True Positives:  {tp} (correctly identified phishing)")
    print(f"  True Negatives:  {tn} (correctly identified legitimate)")
    print(f"  False Positives: {fp} (legitimate flagged as phishing)")
    print(f"  False Negatives: {fn} (phishing missed)")
    
    # Classification Report
    print_section("DETAILED CLASSIFICATION REPORT")
    print()
    print(classification_report(y_test, y_pred, 
                                target_names=['Legitimate', 'Phishing'],
                                digits=4))
    
    # Cross-Validation
    print_section("CROSS-VALIDATION (10-FOLD)")
    
    print("\nPerforming 10-fold cross-validation...")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    cv_accuracy = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    cv_precision = cross_val_score(model, X, y, cv=skf, scoring='precision')
    cv_recall = cross_val_score(model, X, y, cv=skf, scoring='recall')
    cv_f1 = cross_val_score(model, X, y, cv=skf, scoring='f1')
    
    print(f"\n  Accuracy:  {cv_accuracy.mean():.4f} ± {cv_accuracy.std():.4f}")
    print(f"  Precision: {cv_precision.mean():.4f} ± {cv_precision.std():.4f}")
    print(f"  Recall:    {cv_recall.mean():.4f} ± {cv_recall.std():.4f}")
    print(f"  F1-Score:  {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
    
    # Feature Importance
    print_section("FEATURE IMPORTANCE")
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = [f'F{i+1}' for i in range(12)]
        
        feature_desc = {
            'F1': 'Brand Similarity',
            'F2': 'Subdomain Depth',
            'F3': 'Free CA',
            'F4': 'Suspicious TLD',
            'F5': 'Inner TLD',
            'F6': 'Suspicious Keywords',
            'F7': 'High Entropy',
            'F8': 'Hyphens',
            'F9': 'SAN Analysis',
            'F10': 'Self-Signed Cert',
            'F11': 'Validity Period',
            'F12': 'Chain Validation'
        }
        
        print("\n  Rank  Feature  Importance  Description")
        print("  " + "-"*60)
        
        sorted_features = sorted(zip(feature_names, importances), 
                                key=lambda x: x[1], reverse=True)
        
        for rank, (name, imp) in enumerate(sorted_features, 1):
            desc = feature_desc.get(name, '')
            print(f"  {rank:2d}.   {name:4s}     {imp:6.4f}     {desc}")
        
        # Group by category
        print("\n  Feature Category Importance:")
        domain_features = sum(importances[:8])
        cert_features = sum(importances[8:12])
        
        print(f"    Domain Features (F1-F8):  {domain_features:.4f} ({domain_features*100:.1f}%)")
        print(f"    Cert Features (F9-F12):   {cert_features:.4f} ({cert_features*100:.1f}%)")
    
    # Summary
    print_section("SUMMARY")
    
    print(f"\n  ✓ Model: {type(model).__name__}")
    print(f"  ✓ Accuracy: {accuracy*100:.2f}%")
    print(f"  ✓ Precision: {precision*100:.2f}%")
    print(f"  ✓ Recall: {recall*100:.2f}%")
    print(f"  ✓ F1-Score: {f1*100:.2f}%")
    print(f"  ✓ Features: 12 (F1-F12)")
    print(f"  ✓ Test Samples: {len(X_test)}")
    
    baseline_accuracy = 0.8752
    improvement = (accuracy - baseline_accuracy) * 100
    
    if improvement > 0:
        print(f"\n  🎯 Improvement over baseline: +{improvement:.2f}%")
    else:
        print(f"\n  ⚠️  Performance vs baseline: {improvement:.2f}%")
    
    print("\n" + "="*70)
    print()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm
    }


if __name__ == '__main__':
    try:
        metrics = evaluate_model()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("  1. training_data.npz (run: python prepare_data.py)")
        print("  2. phishhook_model.pkl (run: python train.py)")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
