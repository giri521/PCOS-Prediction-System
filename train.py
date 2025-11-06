# ==============================================================
# 🩺 PCOS Prediction Model Training (3 Separate Models)
# Author: Giri Vennapusa
# ==============================================================
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.metrics import accuracy_score, f1_score

# --- Configuration ---
TARGET_COL = 'PCOS (Y/N)'
MODELS_DIR = "models"
DATASET_PATH = "dataset/pcos_data.csv"

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

try:
    # ========== Step 1: Load dataset and preprocess ==========
    data = pd.read_csv(DATASET_PATH)
    data = data.dropna(axis=1, thresh=len(data) * 0.7)
    data = data.fillna(data.median(numeric_only=True))

    if TARGET_COL not in data.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset")

    X = data.drop(columns=[TARGET_COL])
    y = data[TARGET_COL]

    # Encode Y/N to 0/1
    for col in X.select_dtypes(include='object').columns:
        X[col] = X[col].replace({'Y': 1, 'N': 0, 'Yes': 1, 'No': 0}).astype(float)

    # Rename Follicle columns
    if 'Follicle No. (L)' in X.columns:
        X.rename(columns={'Follicle No. (L)': 'Ls', 'Follicle No. (R)': 'Rs'}, inplace=True)

    # ========== Step 2: Train-Test Split ==========
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # ========== Step 3: Scaling ==========
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)

    # Save full columns and scaler (used for all models)
    joblib.dump(X_train.columns.tolist(), os.path.join(MODELS_DIR, "full_columns.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))

    # ========== Step 4: Feature Selection ==========
    mi_scores = mutual_info_classif(X_train_scaled, y_train, random_state=42)
    mi_features = list(X_train.columns[np.argsort(mi_scores)[-10:]])

    f_scores, _ = f_classif(X_train_scaled, y_train)
    ttest_features = list(X_train.columns[np.argsort(f_scores)[-10:]])

    combined_features = list(set(mi_features) | set(ttest_features))

    # Save selected feature lists
    joblib.dump(mi_features, os.path.join(MODELS_DIR, "mi_features.joblib"))
    joblib.dump(ttest_features, os.path.join(MODELS_DIR, "t_features.joblib"))
    joblib.dump(combined_features, os.path.join(MODELS_DIR, "combined_features.joblib"))

    print("✅ Feature selection completed successfully.")
    print("Top MI Features:", mi_features)
    print("Top T-Test Features:", ttest_features)
    print("Combined Features:", combined_features)

    # ========== Step 5: Define a function to train and evaluate ==========
    def train_and_save_model(feature_set, model_name):
        X_tr = X_train_scaled[feature_set]
        X_te = X_test_scaled[feature_set]

        clf = BaggingClassifier(
            RandomForestClassifier(n_estimators=100, random_state=42),
            n_estimators=10, random_state=42, n_jobs=-1
        )
        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)

        acc = accuracy_score(y_test, y_pred) * 100
        f1 = f1_score(y_test, y_pred, average='binary') * 100

        print(f"\n📦 {model_name} Model Evaluation:")
        print(f"Accuracy: {acc:.2f}%")
        print(f"F1-Score: {f1:.2f}%")

        # Save model
        model_path = os.path.join(MODELS_DIR, f"{model_name}_model.joblib")
        joblib.dump(clf, model_path)

        return acc, f1

    # ========== Step 6: Train 3 Models ==========
    results = {}
    results['mi'] = train_and_save_model(mi_features, "mi")
    results['ttest'] = train_and_save_model(ttest_features, "ttest")
    results['combined'] = train_and_save_model(combined_features, "combined")

    # ========== Step 7: Display Summary ==========
    print("\n=================== SUMMARY ===================")
    for key, (acc, f1) in results.items():
        print(f"{key.upper()} Model -> Accuracy: {acc:.2f}%, F1: {f1:.2f}%")
    print("===============================================")
    print(f"✅ All models and scalers saved to '{MODELS_DIR}' folder.")

except FileNotFoundError:
    print(f"\n❌ ERROR: Dataset not found at '{DATASET_PATH}'. Please check file path.")
except Exception as e:
    print(f"\n❌ An error occurred during model training: {e}")
