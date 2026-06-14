import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Model Imports
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier 

def train_model(matches_df, features, target_col="target", verbose=True):

    X = matches_df[features]
    y = matches_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression (L1)": LogisticRegression(
            solver="saga",          
            l1_ratio=1.0, 
            C=0.5,
            max_iter=5000, 
            class_weight="balanced",
            random_state=42
        ),
        "Support Vector Machine (SVM)": SVC(
            kernel="rbf",
            C=1.0,
            class_weight="balanced",
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=5,         
            learning_rate=0.05,
            random_state=42
        )
    }

    results = {}
    best_accuracy = 0
    best_model_name = None

    for name, model in models.items():
        print("\n" + "="*50)
        print(f"TRAINING MODEL: {name}")
        print("="*50)
        
        if name == "XGBoost":
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        else:
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, preds)
        results[name] = acc
        
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds))

        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            
            joblib.dump(model, "models/football_model.pkl")
            joblib.dump(scaler, "models/scaler.pkl")

    if verbose:
        print("\n" + "#"*50)
        print("FINAL MODEL COMPARISON SUMMARY")
        print("#"*50)
        for name, acc in results.items():
            print(f"- {name}: {acc:.4f} Accuracy")
        print(f"\nSaved the top-performing model ({best_model_name}) to 'models/football_model.pkl'")

    return models[best_model_name]