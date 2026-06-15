import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Model Imports
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

def train_model(matches_df, features, target_col="target", verbose=True):
    # ==========================================
    # DIAGNOSTIC LOG 1: DATASET INSPECTION
    # ==========================================
    print("\n[DEBUG LOG] Starting Data Diagnostics...")
    print(f"-> Total records available for training/testing: {len(matches_df)}")
    
    # Check for chronological sorting issues
    if 'date' in matches_df.columns:
        print(f"-> Data Date Range: {matches_df['date'].min()} to {matches_df['date'].max()}")
        # Check if dataset is truly sorted by date before chronological split
        is_sorted = matches_df['date'].is_monotonic_increasing
        print(f"-> Is dataset chronologically ordered? {is_sorted}")
        if not is_sorted:
            print("   ⚠️ WARNING: Dataset is not sorted by date. Sorting now to avoid data leakage.")
            matches_df = matches_df.sort_values('date').reset_index(drop=True)

    X = matches_df[features]
    y = matches_df[target_col]

    # Chronological Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # ==========================================
    # DIAGNOSTIC LOG 2: SPLIT SEPARABILITY
    # ==========================================
    if 'date' in matches_df.columns:
        train_idx = X_train.index
        test_idx = X_test.index
        print(f"-> Train Set Splits: {matches_df.loc[train_idx, 'date'].min()} to {matches_df.loc[train_idx, 'date'].max()} ({len(X_train)} matches)")
        print(f"-> Test Set Splits:  {matches_df.loc[test_idx, 'date'].min()} to {matches_df.loc[test_idx, 'date'].max()} ({len(X_test)} matches)")

    print("\n-> Target distribution in Training set:")
    print(y_train.value_counts(normalize=True).to_string())
    print("-> Target distribution in Test set:")
    print(y_test.value_counts(normalize=True).to_string())

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression (L1)": LogisticRegression(
            solver="saga",          
            penalty="l1", 
            C=0.5,
            max_iter=5000, 
            class_weight="balanced",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, 
            max_depth=6, 
            min_samples_leaf=5, 
            class_weight="balanced", 
            random_state=42, 
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150, 
            max_depth=4, 
            learning_rate=0.03, 
            subsample=0.8, 
            colsample_bytree=0.8,
            objective="binary:logistic",  # Use binary classification
            # ❌ REMOVE num_class=3 from here!
            random_state=42, 
            n_jobs=-1
        )
    }

    if LGBMClassifier is not None:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=150, 
            max_depth=4, 
            learning_rate=0.03, 
            objective="binary",          # Change from multiclass to binary
            class_weight="balanced", 
            random_state=42, 
            n_jobs=-1, 
            verbose=-1
        )

    results = {}
    best_accuracy = 0
    best_model_name = None
    best_model_instance = None
    tree_based_models = ["Random Forest", "XGBoost", "LightGBM"]

    for name, model in models.items():
        if name in tree_based_models:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        else:
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, preds)
        results[name] = acc

        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            best_model_instance = model
            joblib.dump(model, "models/football_model.pkl")
            joblib.dump(scaler, "models/scaler.pkl")

    # ==========================================
    # DIAGNOSTIC LOG 3: FEATURE IMPORTANCE
    # ==========================================
    print("\n=========================================")
    print(f"[DIAGNOSTIC LOG] Feature Importance Profiles ({best_model_name})")
    print("=========================================")
    
    if hasattr(best_model_instance, 'feature_importances_'):
        importances = best_model_instance.feature_importances_
        feat_imp = pd.DataFrame({'Feature': features, 'Importance': importances})
        feat_imp = feat_imp.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        print(feat_imp.to_string())
    elif hasattr(best_model_instance, 'coef_'):
        importances = np.abs(best_model_instance.coef_[0])
        feat_imp = pd.DataFrame({'Feature': features, 'Importance': importances})
        feat_imp = feat_imp.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        print(feat_imp.to_string())

    print("\n=========================================")
    print("FINAL BENCHMARK SUMMARY")
    print("=========================================")
    if verbose:
        for name, acc in results.items():
            marker = "⭐" if name == best_model_name else "-"
            print(f"{marker} {name}: {acc:.4f} Accuracy")
    print(f"\n🏆 Best Model: {best_model_name} ({best_accuracy:.4f} Accuracy)")

    return best_model_instance