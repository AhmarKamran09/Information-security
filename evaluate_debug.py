
import pickle
import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
from uci_mapper import prepare_training_data

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='uci-ml-phishing-dataset.csv')
    args = parser.parse_args()
    
    print(f"Loading model from {args.model}...")
    with open(args.model, 'rb') as f:
        model_data = pickle.load(f)
    
    if isinstance(model_data, dict) and 'model' in model_data:
        model = model_data['model']
        scaler = model_data.get('scaler', None)
    else:
        model = model_data
        scaler = None
        
    print(f"Loading dataset from {args.dataset}...")
    X, y = prepare_training_data(args.dataset)
    
    # Pad with zeros for F9-F12
    X_padded = np.hstack([X, np.zeros((X.shape[0], 4))])
    
    from sklearn.model_selection import train_test_split
    # Use same random state as training script
    X_train, X_test, y_train, y_test = train_test_split(
        X_padded, y, test_size=0.2, random_state=42, stratify=y
    )
    
    if scaler is not None:
        X_test = scaler.transform(X_test)
        
    results = evaluate_model(model, X_test, y_test)
    
    print(f"Model: {args.model}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")

if __name__ == '__main__':
    main()
