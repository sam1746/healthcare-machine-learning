import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, classification_report

def prepare_data():
    csv_file = "diabetes_data.csv"
    if os.path.exists(csv_file):
        print(f"Loading local dataset: {csv_file}")
        df = pd.read_csv(csv_file)
    else:
        print("Dataset not found. Generating realistic clinical dataset...")
        np.random.seed(42)
        n = 3000

        age = np.random.randint(20, 80, size=n)
        bmi = np.round(np.random.normal(28, 6, size=n), 1)
        glucose = np.random.normal(115, 35, size=n)
        hba1c = np.round(np.random.normal(5.8, 1.2, size=n), 1)
        systolic_bp = np.random.normal(125, 18, size=n).astype(int)
        cholesterol = np.random.normal(200, 40, size=n).astype(int)
        hypertension = (systolic_bp > 135).astype(int)
        heart_disease = np.random.binomial(1, 0.08, size=n)

        logit = (
            -9.5
            + 0.03 * age
            + 0.08 * bmi
            + 0.025 * glucose
            + 0.75 * hba1c
            + 0.5 * hypertension
            + 0.6 * heart_disease
        )
        prob = 1 / (1 + np.exp(-logit))
        target = np.random.binomial(1, prob)

        df = pd.DataFrame({
            "Age": age,
            "BMI": np.clip(bmi, 15.0, 50.0),
            "Glucose": np.clip(glucose, 70, 300).astype(int),
            "HbA1c": np.clip(hba1c, 4.0, 14.0),
            "SystolicBP": np.clip(systolic_bp, 90, 200),
            "Cholesterol": np.clip(cholesterol, 120, 350),
            "Hypertension": hypertension,
            "HeartDisease": heart_disease,
            "Outcome": target
        })
        df.to_csv(csv_file, index=False)
        print("Generated and saved 'diabetes_data.csv'.")
    return df

def main():
    df = prepare_data()

    feature_cols = [
        "Age", "BMI", "Glucose", "HbA1c", 
        "SystolicBP", "Cholesterol", "Hypertension", "HeartDisease"
    ]
    X = df[feature_cols]
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Logistic Regression
    lr = LogisticRegression(class_weight="balanced", random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_preds = lr.predict(X_test_scaled)
    lr_probs = lr.predict_proba(X_test_scaled)[:, 1]

    # Random Forest
    rf = RandomForestClassifier(n_estimators=150, max_depth=6, class_weight="balanced", random_state=42)
    rf.fit(X_train_scaled, y_train)
    rf_preds = rf.predict(X_test_scaled)
    rf_probs = rf.predict_proba(X_test_scaled)[:, 1]

    print("\n" + "=" * 50)
    print("MODEL COMPARISON ON TEST SET")
    print("=" * 50)
    print(f"Logistic Regression -> Accuracy: {accuracy_score(y_test, lr_preds):.3f} | Recall: {recall_score(y_test, lr_preds):.3f} | ROC-AUC: {roc_auc_score(y_test, lr_probs):.3f}")
    print(f"Random Forest       -> Accuracy: {accuracy_score(y_test, rf_preds):.3f} | Recall: {recall_score(y_test, rf_preds):.3f} | ROC-AUC: {roc_auc_score(y_test, rf_probs):.3f}")
    print("\nDetailed Random Forest Report:\n", classification_report(y_test, rf_preds))

    model_payload = {
        "model": rf,
        "scaler": scaler,
        "features": feature_cols
    }
    joblib.dump(model_payload, "health_risk_model.pkl")
    print("Artifact saved successfully as 'health_risk_model.pkl'")

if __name__ == "__main__":
    main()