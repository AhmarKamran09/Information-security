"""
Phish-Hook Model Evaluation Script
Evaluates trained models and generates performance reports.
"""

import pickle
import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
from uci_mapper import prepare_training_data


def load_model(model_path: str):
    """Load trained model from pickle file."""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # Handle both formats: dict (advanced) or direct model (standard)
    if isinstance(model_data, dict) and 'model' in model_data:
        return model_data['model']
    else:
        return model_data


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # ROC curve if probabilities available
    roc_auc = None
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }


def print_evaluation_report(results: dict, model_name: str = "Model"):
    """Print detailed evaluation report."""
    print("\n" + "="*60)
    print(f"{model_name} Evaluation Report")
    print("="*60)
    
    print("\nPerformance Metrics:")
    print(f"  Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"  Precision: {results['precision']:.4f} ({results['precision']*100:.2f}%)")
    print(f"  Recall:    {results['recall']:.4f} ({results['recall']*100:.2f}%)")
    print(f"  F1 Score:  {results['f1']:.4f} ({results['f1']*100:.2f}%)")
    
    if results['roc_auc']:
        print(f"  ROC AUC:   {results['roc_auc']:.4f}")
    
    print("\nConfusion Matrix:")
    cm = results['confusion_matrix']
    print(f"                Predicted")
    print(f"              Legit  Phishing")
    print(f"Actual Legit    {cm[0][0]:4d}    {cm[0][1]:4d}")
    print(f"      Phishing {cm[1][0]:4d}    {cm[1][1]:4d}")
    
    print("\nPaper Target Performance:")
    print("  Accuracy:  ~95%")
    print("  Precision: ~93-96%")
    print("  Recall:    ~93-95%")
    print("  F1 Score:  ~94-95%")
    
    print("\nComparison:")
    accuracy_match = "✅" if results['accuracy'] >= 0.93 else "❌"
    precision_match = "✅" if 0.93 <= results['precision'] <= 0.97 else "❌"
    recall_match = "✅" if 0.93 <= results['recall'] <= 0.96 else "❌"
    f1_match = "✅" if 0.94 <= results['f1'] <= 0.96 else "❌"
    
    print(f"  Accuracy:  {accuracy_match} {'Match' if results['accuracy'] >= 0.93 else 'Below target'}")
    print(f"  Precision: {precision_match} {'Match' if 0.93 <= results['precision'] <= 0.97 else 'Outside range'}")
    print(f"  Recall:    {recall_match} {'Match' if 0.93 <= results['recall'] <= 0.96 else 'Outside range'}")
    print(f"  F1 Score:  {f1_match} {'Match' if 0.94 <= results['f1'] <= 0.96 else 'Outside range'}")


def plot_roc_curve(y_test, y_pred_proba, output_file: str = None):
    """Plot ROC curve."""
    if y_pred_proba is None:
        print("Cannot plot ROC curve: model does not support probability prediction")
        return
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    if output_file:
        plt.savefig(output_file)
        print(f"\nROC curve saved to {output_file}")
    else:
        plt.show()


def cross_validate(model, X, y, cv=10):
    """Perform cross-validation."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    cv_scores = {
        'accuracy': cross_val_score(model, X, y, cv=skf, scoring='accuracy'),
        'precision': cross_val_score(model, X, y, cv=skf, scoring='precision'),
        'recall': cross_val_score(model, X, y, cv=skf, scoring='recall'),
        'f1': cross_val_score(model, X, y, cv=skf, scoring='f1')
    }
    
    print("\n" + "="*60)
    print(f"{cv}-Fold Cross-Validation Results")
    print("="*60)
    
    for metric_name, scores in cv_scores.items():
        mean_score = scores.mean()
        std_score = scores.std()
        print(f"\n{metric_name.capitalize()}:")
        print(f"  Mean: {mean_score:.4f}")
        print(f"  Std:  {std_score:.4f}")
        print(f"  Range: [{mean_score - 2*std_score:.4f}, {mean_score + 2*std_score:.4f}]")
    
    return cv_scores


def main():
    parser = argparse.ArgumentParser(description='Evaluate Phish-Hook model')
    parser.add_argument('--model', type=str, default='phishhook_model.pkl',
                       help='Path to trained model')
    parser.add_argument('--dataset', type=str, default='uci-ml-phishing-dataset.csv',
                       help='Path to UCI dataset')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size')
    parser.add_argument('--cv', type=int, default=10,
                       help='Number of cross-validation folds')
    parser.add_argument('--plot-roc', action='store_true',
                       help='Plot ROC curve')
    parser.add_argument('--roc-output', type=str, default='roc_curve.png',
                       help='Output file for ROC curve')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model}...")
    model_data = pickle.load(open(args.model, 'rb'))
    
    # Handle both formats: dict (advanced) or direct model (standard)
    if isinstance(model_data, dict) and 'model' in model_data:
        model = model_data['model']
        scaler = model_data.get('scaler', None)
        print(f"Model type: {model_data.get('model_name', 'Unknown')}")
    else:
        model = model_data
        scaler = None
    
    # Load data
    print(f"Loading dataset from {args.dataset}...")
    X, y = prepare_training_data(args.dataset)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    
    print(f"\nTest set size: {len(X_test)} samples")
    
    # Apply scaler if available
    if scaler is not None:
        print("Applying scaler to test data...")
        X_test = scaler.transform(X_test)
    
    # Evaluate model
    results = evaluate_model(model, X_test, y_test)
    print_evaluation_report(results, "Phish-Hook Model")
    
    # Cross-validation
    if args.cv > 0:
        cv_results = cross_validate(model, X_train, y_train, cv=args.cv)
    
    # Plot ROC curve
    if args.plot_roc:
        plot_roc_curve(y_test, results['y_pred_proba'], args.roc_output)
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)


if __name__ == '__main__':
    main()

