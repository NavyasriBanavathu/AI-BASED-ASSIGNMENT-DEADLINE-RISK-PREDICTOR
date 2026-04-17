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

# Test no noise
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
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test_text, rf_pred)

xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9, use_label_encoder=False, eval_metric="logloss", random_state=42)
xgb.fit(X_train, y_train_enc)
xgb_pred = le.inverse_transform(xgb.predict(X_test))
xgb_acc = accuracy_score(y_test_text, xgb_pred)

print(f"No noise: RF={rf_acc:.4f}, XGB={xgb_acc:.4f}")

# Test with seed 42
X_seed = df[["avg_late_days", "gpa", "absences"]].copy()
y_text_seed = df["risk_label"].copy()

np.random.seed(42)
X_seed["avg_late_days"] += np.random.randint(0, 2, size=len(X_seed))
X_seed["absences"] += np.random.randint(0, 2, size=len(X_seed))
X_seed["gpa"] += np.random.normal(0, 0.05, len(X_seed))
X_seed["avg_late_days"] = X_seed["avg_late_days"].clip(lower=0).astype(int)
X_seed["absences"] = X_seed["absences"].clip(lower=0).astype(int)
X_seed["gpa"] = X_seed["gpa"].clip(lower=0.0).astype(float)
flip_ratio = 0.02
flip_idx = np.random.choice(len(y_text_seed), int(len(y_text_seed)*flip_ratio), replace=False)
for i in flip_idx:
    y_text_seed.iloc[i] = "Low Risk" if y_text_seed.iloc[i] == "High Risk" else "High Risk"

y_enc_seed = le.transform(y_text_seed)
X_train, X_test, y_train_text, y_test_text, y_train_enc, y_test_enc = train_test_split(
    X_seed, y_text_seed, y_enc_seed, test_size=0.25, random_state=42, stratify=y_text_seed)

rf.fit(X_train, y_train_text)
rf_acc_seed = accuracy_score(y_test_text, rf.predict(X_test))
xgb.fit(X_train, y_train_enc)
xgb_acc_seed = accuracy_score(y_test_text, le.inverse_transform(xgb.predict(X_test)))
print(f"Seed 42: RF={rf_acc_seed:.4f}, XGB={xgb_acc_seed:.4f}")
