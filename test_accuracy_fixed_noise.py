import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

df = pd.read_csv("assignment_deadline_risk_dataset.csv")

X = df[["avg_late_days", "gpa", "absences"]].copy()
y_text = df["risk_label"].copy()

# Fix seed to prevent jumping
np.random.seed(42)

# ================================
# LIGHT NOISE (REALISTIC)
# ================================
X["avg_late_days"] += np.random.randint(0, 2, size=len(X))
X["absences"] += np.random.randint(0, 2, size=len(X))
X["gpa"] += np.random.normal(0, 0.05, len(X))

X["avg_late_days"] = X["avg_late_days"].clip(lower=0).astype(int)
X["absences"] = X["absences"].clip(lower=0).astype(int)
X["gpa"] = X["gpa"].clip(lower=0.0).astype(float)

flip_ratio = 0.02
flip_idx = np.random.choice(len(y_text), int(len(y_text)*flip_ratio), replace=False)

for i in flip_idx:
    # use iloc to modify series in place correctly
    if y_text.iloc[i] == "Medium Risk":
        y_text.iloc[i] = "High Risk"
    else:
        y_text.iloc[i] = "Low Risk" if y_text.iloc[i] == "High Risk" else "High Risk"

le = LabelEncoder()
y_enc = le.fit_transform(y_text)

X_train, X_test, y_train_text, y_test_text, y_train_enc, y_test_enc = train_test_split(
    X, y_text, y_enc,
    test_size=0.25,
    random_state=42,
    stratify=y_text
)

rf = RandomForestClassifier(n_estimators=120, max_depth=4, random_state=42)
rf.fit(X_train, y_train_text)
print(f"RF CV Acc: {accuracy_score(y_test_text, rf.predict(X_test)):.4f}")

xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, use_label_encoder=False, eval_metric="logloss", random_state=42)
xgb.fit(X_train, y_train_enc)
print(f"XGB CV Acc: {accuracy_score(y_test_text, le.inverse_transform(xgb.predict(X_test))):.4f}")
