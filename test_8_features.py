
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from uci_mapper import prepare_training_data

def main():
    print("Loading data...")
    X, y = prepare_training_data('uci-ml-phishing-dataset.csv')
    
    # NO PADDING - Use pure 8 features
    print(f"Features shape: {X.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    models = {
        'DT': DecisionTreeClassifier(random_state=42),
        'RF': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"{name} (8 features): Accuracy={acc:.4f}")

if __name__ == '__main__':
    main()
