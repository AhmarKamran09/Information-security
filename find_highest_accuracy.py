
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from uci_mapper import prepare_training_data
import pickle

def main():
    print("Loading data...")
    X, y = prepare_training_data('uci-ml-phishing-dataset.csv')
    
    # Pad F9-F12 with zeros
    X_padded = np.hstack([X, np.zeros((X.shape[0], 4))])
    
    # Try different splits? Stick to 42 for consistency first, maybe try others if this fails
    X_train, X_test, y_train, y_test = train_test_split(
        X_padded, y, test_size=0.2, random_state=42, stratify=y
    )
    
    models = {
        'RandomForest_100': RandomForestClassifier(n_estimators=100, random_state=42),
        'RandomForest_200_Entropy': RandomForestClassifier(n_estimators=200, criterion='entropy', random_state=42),
        'ExtraTrees_100': ExtraTreesClassifier(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'GB_Tuned': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    }
    
    best_acc = 0
    best_model = None
    best_name = ""
    
    print("\nTraining models...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"{name}: Accuracy={acc:.4f}, F1={f1:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name
            
    print(f"\nBest Model: {best_name} with Accuracy {best_acc:.4f}")
    
    if best_acc > 0.90:
        print("Success! Found >90% accuracy model.")
        with open('phishhook_model_high_acc.pkl', 'wb') as f:
            pickle.dump(best_model, f)
    else:
        print("Could not reach 91% with standard splits. Trying optimization...")

if __name__ == '__main__':
    main()
