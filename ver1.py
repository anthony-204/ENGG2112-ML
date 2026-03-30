import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    roc_curve, roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, classification_report
)

# Load data
df = pd.read_csv("ev_battery_charging_data.csv")

# Create binary target: maintenance needed
# Adjust threshold depending on your engineering interpretation
threshold_deg = 10.0
df["maintenance_needed"] = (df["Degradation Rate (%)"] > threshold_deg).astype(int)

# Features and target
X = df.drop(columns=["maintenance_needed"])
y = df["maintenance_needed"]

# Remove columns that leak the target if needed
# Since degradation directly defines the label, do NOT include it as a feature
X = X.drop(columns=["Degradation Rate (%)"])

# Identify categorical and numerical columns
categorical_cols = ["Charging Mode", "Battery Type", "EV Model"]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# Train / validation / test split (60 / 20 / 20)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print("Train:", X_train.shape, "Validation:", X_val.shape, "Test:", X_test.shape)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42),
}

results = {}

for name, model in models.items():
    clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])
    
    clf.fit(X_train, y_train)
    
    # Validation probabilities
    y_val_proba = clf.predict_proba(X_val)[:, 1]
    y_val_pred = (y_val_proba >= 0.5).astype(int)

    acc = accuracy_score(y_val, y_val_pred)
    prec = precision_score(y_val, y_val_pred, zero_division=0)
    rec = recall_score(y_val, y_val_pred, zero_division=0)
    f1 = f1_score(y_val, y_val_pred, zero_division=0)
    auc = roc_auc_score(y_val, y_val_proba)

    results[name] = {
        "model": clf,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
    }

for name, metrics in results.items():
    print(f"\n{name}")
    print(f"Accuracy:  {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1-score:  {metrics['f1']:.3f}")
    print(f"AUC:       {metrics['auc']:.3f}")



# Choose the best model by validation AUC
best_name = max(results, key=lambda k: results[k]["auc"])
best_model = results[best_name]["model"]
print("Best model:", best_name)

# Validation ROC curve
y_val_proba = best_model.predict_proba(X_val)[:, 1]
fpr, tpr, thresholds = roc_curve(y_val, y_val_proba)
auc = roc_auc_score(y_val, y_val_proba)

print("AUC:", auc)

# Find threshold closest to the top-left corner
dist = np.sqrt((fpr - 0)**2 + (1 - tpr)**2)
best_idx = np.argmin(dist)
best_threshold = thresholds[best_idx]

print("Best threshold:", best_threshold)
print("TPR at best threshold:", tpr[best_idx])
print("FPR at best threshold:", fpr[best_idx])

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Fit best model on train + validation, then test
X_train_full = pd.concat([X_train, X_val], axis=0)
y_train_full = pd.concat([y_train, y_val], axis=0)

best_model.fit(X_train_full, y_train_full)

y_test_proba = best_model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_proba >= best_threshold).astype(int)

test_acc = accuracy_score(y_test, y_test_pred)
test_prec = precision_score(y_test, y_test_pred, zero_division=0)
test_rec = recall_score(y_test, y_test_pred, zero_division=0)
test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
test_auc = roc_auc_score(y_test, y_test_proba)

print("\nTEST RESULTS")
print("Accuracy:", round(test_acc, 3))
print("Precision:", round(test_prec, 3))
print("Recall / TPR:", round(test_rec, 3))
print("F1-score:", round(test_f1, 3))
print("AUC:", round(test_auc, 3))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))