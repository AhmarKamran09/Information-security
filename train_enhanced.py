"""
Train Enhanced Phish-Hook Model
Trains with F1-F12 features for improved accuracy.
"""

import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from uci_mapper import get_class_distribution


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
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


print("="*60)
print("Phish-Hook Enhanced Model Training")
print("="*60)

# Load data
print("\nLoading enhanced data...")
data = np.load('uci_enhanced.npz')
X = data['X']
y = data['y']

print(f"  Samples: {len(X)}")
print(f"  Features: {X.shape[1]} (F1-F12)")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain/Test split:")
print(f"  Training: {len(X_train)}")
print(f"  Test: {len(X_test)}")

# Balance
print("\n" + "="*60)
print("Class Balancing")
print("="*60)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

undersampler = RandomUnderSampler(random_state=42)
X_train_balanced, y_train_balanced = undersampler.fit_resample(X_train_smote, y_train_smote)

print(f"\nBalanced: {len(X_train_balanced)} samples")

# Train
print("\n" + "="*60)
print("Training Random Forest")
print("="*60)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

print("\nTraining...")
model.fit(X_train_balanced, y_train_balanced)

# Evaluate
y_pred = model.predict(X_test)
metrics = print_metrics(y_test, y_pred, "Random Forest")

# Feature importance
print("\n" + "="*60)
print("Feature Importance")
print("="*60)

importances = model.feature_importances_
feature_names = [f'F{i+1}' for i in range(12)]

print("\nTop Features:")
for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:8]:
    print(f"  {name}: {imp:.4f} ({imp*100:.1f}%)")

# Save
print(f"\nSaving model to phishhook_model_final.pkl...")
with open('phishhook_model_final.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\n" + "="*60)
print("Training Complete!")
print("="*60)
print(f"\nFinal Accuracy: {metrics['accuracy']*100:.2f}%")
print(f"Model saved: phishhook_model_final.pkl")
