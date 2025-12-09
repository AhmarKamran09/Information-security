"""
Phish-Hook Model Training Script
Trains SVM and other ML models on UCI dataset with SMOTE and undersampling.
"""

import numpy as np
import pandas as pd
import pickle
import argparse
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from uci_mapper import prepare_training_data, get_class_distribution


def print_metrics(y_true, y_pred, model_name="Model"):
    """Print evaluation metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    
    print(f"\n{model_name} Performance:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def train_models(X_train, y_train, X_test, y_test):
    """
    Train multiple models and return the best one.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary of trained models and their metrics
    """
    models = {
        'SVM': SVC(kernel='linear', C=0.03, probability=True, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }
    
    results = {}
    
    print("\n" + "="*60)
    print("Training Models")
    print("="*60)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        metrics = print_metrics(y_test, y_pred, name)
        results[name] = {
            'model': model,
            'metrics': metrics
        }
    
    return results


def cross_validate_model(model, X, y, cv=10):
    """
    Perform k-fold cross-validation.
    
    Args:
        model: Model to validate
        X: Features
        y: Labels
        cv: Number of folds
        
    Returns:
        Dictionary with CV metrics
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    cv_accuracy = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    cv_precision = cross_val_score(model, X, y, cv=skf, scoring='precision')
    cv_recall = cross_val_score(model, X, y, cv=skf, scoring='recall')
    cv_f1 = cross_val_score(model, X, y, cv=skf, scoring='f1')
    
    return {
        'accuracy': {
            'mean': cv_accuracy.mean(),
            'std': cv_accuracy.std()
        },
        'precision': {
            'mean': cv_precision.mean(),
            'std': cv_precision.std()
        },
        'recall': {
            'mean': cv_recall.mean(),
            'std': cv_recall.std()
        },
        'f1': {
            'mean': cv_f1.mean(),
            'std': cv_f1.std()
        }
    }


def apply_smote_and_undersampling(X, y):
    """
    Apply SMOTE oversampling and random undersampling.
    
    Args:
        X: Features
        y: Labels
        
    Returns:
        Resampled X and y
    """
    print("\n" + "="*60)
    print("Class Balancing")
    print("="*60)
    
    # Print original distribution
    print("\nOriginal class distribution:")
    dist = get_class_distribution(y)
    for label, stats in dist.items():
        label_name = "Phishing/Suspicious" if label == 1 else "Legitimate"
        print(f"  {label_name}: {stats['count']} ({stats['percentage']:.2f}%)")
    
    # Apply SMOTE (oversample minority class)
    print("\nApplying SMOTE oversampling...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    print("After SMOTE:")
    dist_smote = get_class_distribution(y_resampled)
    for label, stats in dist_smote.items():
        label_name = "Phishing/Suspicious" if label == 1 else "Legitimate"
        print(f"  {label_name}: {stats['count']} ({stats['percentage']:.2f}%)")
    
    # Apply Random Undersampling (undersample majority class)
    print("\nApplying Random Undersampling...")
    undersampler = RandomUnderSampler(random_state=42)
    X_balanced, y_balanced = undersampler.fit_resample(X_resampled, y_resampled)
    
    print("After Undersampling:")
    dist_final = get_class_distribution(y_balanced)
    for label, stats in dist_final.items():
        label_name = "Phishing/Suspicious" if label == 1 else "Legitimate"
        print(f"  {label_name}: {stats['count']} ({stats['percentage']:.2f}%)")
    
    return X_balanced, y_balanced


def main():
    parser = argparse.ArgumentParser(description='Train Phish-Hook models')
    parser.add_argument('--dataset', type=str, default='uci-ml-phishing-dataset.csv',
                       help='Path to UCI phishing dataset CSV')
    parser.add_argument('--model-output', type=str, default='phishhook_model.pkl',
                       help='Output path for trained model')
    parser.add_argument('--cv-folds', type=int, default=10,
                       help='Number of cross-validation folds')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (0.0-1.0)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Phish-Hook Model Training")
    print("="*60)
    
    # Load and prepare data
    print(f"\nLoading dataset from {args.dataset}...")
    X, y = prepare_training_data(args.dataset)
    
    print(f"\nDataset loaded:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {X.shape[1]} (F1-F8)")
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    
    print(f"\nTrain/Test split:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    # Apply SMOTE and undersampling to training data only
    X_train_balanced, y_train_balanced = apply_smote_and_undersampling(X_train, y_train)
    
    # Train models
    results = train_models(X_train_balanced, y_train_balanced, X_test, y_test)
    
    # Find best model (highest F1 score)
    best_model_name = max(results.keys(), key=lambda k: results[k]['metrics']['f1'])
    best_model = results[best_model_name]['model']
    
    print("\n" + "="*60)
    print(f"Best Model: {best_model_name}")
    print("="*60)
    print_metrics(y_test, best_model.predict(X_test), best_model_name)
    
    # Cross-validation on best model
    print("\n" + "="*60)
    print(f"{args.cv_folds}-Fold Cross-Validation ({best_model_name})")
    print("="*60)
    
    # Use balanced training data for CV
    cv_results = cross_validate_model(best_model, X_train_balanced, y_train_balanced, cv=args.cv_folds)
    
    print("\nCross-Validation Results:")
    for metric_name, metric_stats in cv_results.items():
        print(f"  {metric_name.capitalize()}: {metric_stats['mean']:.4f} (+/- {metric_stats['std']*2:.4f})")
    
    # Save best model
    print(f"\nSaving best model to {args.model_output}...")
    with open(args.model_output, 'wb') as f:
        pickle.dump(best_model, f)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nModel saved: {args.model_output}")
    print(f"Best model: {best_model_name}")
    print(f"Expected performance (from paper):")
    print(f"  Accuracy:  ~95%")
    print(f"  Precision: ~93-96%")
    print(f"  Recall:    ~93-95%")
    print(f"  F1 Score:  ~94-95%")


if __name__ == '__main__':
    main()

