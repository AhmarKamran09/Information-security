"""
Advanced Phish-Hook Model Training with Ensemble Methods
Tests Random Forest, XGBoost, and other advanced models for improved accuracy.
"""

import numpy as np
import pandas as pd
import pickle
import argparse
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from uci_mapper import prepare_training_data, get_class_distribution

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")


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


def train_advanced_models(X_train, y_train, X_test, y_test):
    """
    Train advanced ensemble models.
    
    Returns:
        Dictionary of trained models and their metrics
    """
    models = {}
    
    # 1. Random Forest (Optimized)
    print("\n" + "="*60)
    print("Training Random Forest (Optimized)")
    print("="*60)
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    rf_metrics = print_metrics(y_test, y_pred_rf, "Random Forest")
    models['Random Forest'] = {'model': rf_model, 'metrics': rf_metrics}
    
    # 2. Gradient Boosting
    print("\n" + "="*60)
    print("Training Gradient Boosting")
    print("="*60)
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)
    gb_metrics = print_metrics(y_test, y_pred_gb, "Gradient Boosting")
    models['Gradient Boosting'] = {'model': gb_model, 'metrics': gb_metrics}
    
    # 3. XGBoost (if available)
    if XGBOOST_AVAILABLE:
        print("\n" + "="*60)
        print("Training XGBoost")
        print("="*60)
        
        # Calculate scale_pos_weight for imbalance
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        xgb_model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train, y_train)
        y_pred_xgb = xgb_model.predict(X_test)
        xgb_metrics = print_metrics(y_test, y_pred_xgb, "XGBoost")
        models['XGBoost'] = {'model': xgb_model, 'metrics': xgb_metrics}
    
    # 4. Decision Tree (Tuned) - for comparison
    print("\n" + "="*60)
    print("Training Decision Tree (Tuned)")
    print("="*60)
    dt_model = DecisionTreeClassifier(
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42
    )
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    dt_metrics = print_metrics(y_test, y_pred_dt, "Decision Tree (Tuned)")
    models['Decision Tree (Tuned)'] = {'model': dt_model, 'metrics': dt_metrics}
    
    return models


def hyperparameter_tuning_rf(X_train, y_train):
    """
    Perform hyperparameter tuning for Random Forest.
    
    Returns:
        Best Random Forest model
    """
    print("\n" + "="*60)
    print("Hyperparameter Tuning - Random Forest")
    print("="*60)
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [15, 20, 25],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    rf = RandomForestClassifier(
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    print("Running Grid Search (this may take a few minutes)...")
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best F1 Score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def apply_smote_and_undersampling(X, y):
    """Apply SMOTE oversampling and random undersampling."""
    print("\n" + "="*60)
    print("Class Balancing")
    print("="*60)
    
    # Print original distribution
    print("\nOriginal class distribution:")
    dist = get_class_distribution(y)
    for label, stats in dist.items():
        label_name = "Phishing/Suspicious" if label == 1 else "Legitimate"
        print(f"  {label_name}: {stats['count']} ({stats['percentage']:.2f}%)")
    
    # Apply SMOTE
    print("\nApplying SMOTE oversampling...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    print("After SMOTE:")
    dist_smote = get_class_distribution(y_resampled)
    for label, stats in dist_smote.items():
        label_name = "Phishing/Suspicious" if label == 1 else "Legitimate"
        print(f"  {label_name}: {stats['count']} ({stats['percentage']:.2f}%)")
    
    # Apply Random Undersampling
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
    parser = argparse.ArgumentParser(description='Train advanced Phish-Hook models')
    parser.add_argument('--dataset', type=str, default='uci-ml-phishing-dataset.csv',
                       help='Path to UCI phishing dataset CSV')
    parser.add_argument('--model-output', type=str, default='phishhook_model_advanced.pkl',
                       help='Output path for best trained model')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (0.0-1.0)')
    parser.add_argument('--tune', action='store_true',
                       help='Perform hyperparameter tuning (slower but better results)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Advanced Phish-Hook Model Training")
    print("="*60)
    print("\nModels to train:")
    print("  - Random Forest (Optimized)")
    print("  - Gradient Boosting")
    if XGBOOST_AVAILABLE:
        print("  - XGBoost")
    print("  - Decision Tree (Tuned, for comparison)")
    
    # Load and prepare data
    print(f"\nLoading dataset from {args.dataset}...")
    X, y = prepare_training_data(args.dataset)
    
    print(f"\nDataset loaded:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {X.shape[1]} (F1-F8 from UCI dataset)")
    
    # Pad with zeros for F9-F12
    print(f"\nPadding with F9-F12 certificate security features (zeros for UCI data)...")
    X_padded = np.hstack([X, np.zeros((X.shape[0], 4))])
    print(f"  Extended features: {X_padded.shape[1]} (F1-F12)")
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_padded, y, test_size=args.test_size, random_state=42, stratify=y
    )
    
    print(f"\nTrain/Test split:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    # Apply SMOTE and undersampling
    X_train_balanced, y_train_balanced = apply_smote_and_undersampling(X_train, y_train)
    
    # Train models
    models = train_advanced_models(X_train_balanced, y_train_balanced, X_test, y_test)
    
    # Hyperparameter tuning if requested
    if args.tune:
        print("\n" + "="*60)
        print("Hyperparameter Tuning")
        print("="*60)
        best_rf = hyperparameter_tuning_rf(X_train_balanced, y_train_balanced)
        y_pred_tuned = best_rf.predict(X_test)
        tuned_metrics = print_metrics(y_test, y_pred_tuned, "Random Forest (Tuned)")
        models['Random Forest (Tuned)'] = {'model': best_rf, 'metrics': tuned_metrics}
    
    # Find best model
    print("\n" + "="*60)
    print("Model Comparison")
    print("="*60)
    
    print("\n{:<25} {:>10} {:>10} {:>10} {:>10}".format(
        "Model", "Accuracy", "Precision", "Recall", "F1 Score"
    ))
    print("-" * 70)
    
    for name, data in models.items():
        metrics = data['metrics']
        print("{:<25} {:>9.2f}% {:>9.2f}% {:>9.2f}% {:>9.2f}%".format(
            name,
            metrics['accuracy'] * 100,
            metrics['precision'] * 100,
            metrics['recall'] * 100,
            metrics['f1'] * 100
        ))
    
    # Find best model by F1 score
    best_model_name = max(models.keys(), key=lambda k: models[k]['metrics']['f1'])
    best_model = models[best_model_name]['model']
    best_metrics = models[best_model_name]['metrics']
    
    print("\n" + "="*60)
    print(f"Best Model: {best_model_name}")
    print("="*60)
    print_metrics(y_test, best_model.predict(X_test), best_model_name)
    
    # Cross-validation on best model
    print("\n" + "="*60)
    print(f"10-Fold Cross-Validation ({best_model_name})")
    print("="*60)
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_model, X_train_balanced, y_train_balanced, 
                                 cv=skf, scoring='f1')
    
    print(f"\nCross-Validation F1 Scores: {cv_scores}")
    print(f"Mean F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Save best model
    print(f"\nSaving best model to {args.model_output}...")
    with open(args.model_output, 'wb') as f:
        pickle.dump(best_model, f)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nBest Model: {best_model_name}")
    print(f"Model saved: {args.model_output}")
    print(f"\nFinal Performance:")
    print(f"  Accuracy:  {best_metrics['accuracy']*100:.2f}%")
    print(f"  Precision: {best_metrics['precision']*100:.2f}%")
    print(f"  Recall:    {best_metrics['recall']*100:.2f}%")
    print(f"  F1 Score:  {best_metrics['f1']*100:.2f}%")
    
    # Print improvement over baseline
    baseline_accuracy = 0.8752
    improvement = (best_metrics['accuracy'] - baseline_accuracy) * 100
    print(f"\nImprovement over baseline: {improvement:+.2f}%")


if __name__ == '__main__':
    main()
