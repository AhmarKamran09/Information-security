
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from uci_mapper import prepare_training_data

def main():
    X, y = prepare_training_data('uci-ml-phishing-dataset.csv')
    
    # Pad to 12
    X = np.hstack([X, np.zeros((X.shape[0], 4))])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'SVM_RBF': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
        'SVM_RBF_HighC': SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42),
        'MLP_Tuned': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    }
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, model.predict(X_test_scaled))
        print(f"{name}: Accuracy={acc:.4f}")

if __name__ == '__main__':
    main()
