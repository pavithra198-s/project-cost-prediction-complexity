from pathlib import Path
import json, joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report

SEED = 42
np.random.seed(SEED)
BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "Engineering_Cost_Feasibility_Dataset.csv"
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA)

# Cost model: complexity is a legitimate project input when predicting cost.
COST_CATEGORICAL = ["Project_Type", "Project_Name", "Scope_Complexity", "Feasibility_Label"]
COST_NUMERIC = [
    "Resource_Allocation_Score", "Risk_Assessment_Score", "Environmental_Impact_Score",
    "Historical_Cost_Deviation_%", "Stakeholder_Priority_Score", "Time_Estimate_Days"
]
COST_FEATURES = COST_CATEGORICAL + COST_NUMERIC

# Complexity model deliberately excludes Scope_Complexity to avoid target leakage.
COMPLEX_CATEGORICAL = ["Project_Type", "Project_Name", "Feasibility_Label"]
COMPLEX_NUMERIC = COST_NUMERIC
COMPLEX_FEATURES = COMPLEX_CATEGORICAL + COMPLEX_NUMERIC

# Clean types
for c in COST_NUMERIC:
    df[c] = pd.to_numeric(df[c], errors="coerce")
for c in COST_CATEGORICAL:
    df[c] = df[c].astype(str)
df = df.dropna(subset=COST_FEATURES + ["Estimated_Cost_USD", "Scope_Complexity"]).reset_index(drop=True)

# We use one-hot encoding + scaling, then deep neural networks (multi-layer perceptrons).
def make_preprocessor(cat, num):
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ("num", StandardScaler(), num)
    ], remainder="drop")

# ---- Cost deep neural network ----
Xc = df[COST_FEATURES]
yc = np.log1p(df["Estimated_Cost_USD"].astype(float).values)
Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.20, random_state=SEED)
pre_c = make_preprocessor(COST_CATEGORICAL, COST_NUMERIC)
Xc_train_t = pre_c.fit_transform(Xc_train).astype("float32")
Xc_test_t = pre_c.transform(Xc_test).astype("float32")

cost_model = MLPRegressor(hidden_layer_sizes=(256,128,64,32), activation="relu", solver="adam", alpha=0.0005, batch_size=64, learning_rate_init=0.001, max_iter=600, early_stopping=True, validation_fraction=0.15, n_iter_no_change=25, random_state=SEED)
cost_model.fit(Xc_train_t, yc_train)
pred_log = cost_model.predict(Xc_test_t).ravel()
pred_cost = np.expm1(pred_log)
true_cost = np.expm1(yc_test)
cost_mae = float(mean_absolute_error(true_cost, pred_cost))
cost_rmse = float(np.sqrt(mean_squared_error(true_cost, pred_cost)))
cost_r2 = float(r2_score(true_cost, pred_cost))
joblib.dump(cost_model, MODEL_DIR / "cost_predictor.joblib")
joblib.dump(pre_c, MODEL_DIR / "cost_preprocessor.joblib")

# ---- Complexity deep neural network classifier ----
Xq = df[COMPLEX_FEATURES]
labeler = LabelEncoder()
yq = labeler.fit_transform(df["Scope_Complexity"].astype(str))
Xq_train, Xq_test, yq_train, yq_test = train_test_split(Xq, yq, test_size=0.20, random_state=SEED, stratify=yq)
pre_q = make_preprocessor(COMPLEX_CATEGORICAL, COMPLEX_NUMERIC)
Xq_train_t = pre_q.fit_transform(Xq_train).astype("float32")
Xq_test_t = pre_q.transform(Xq_test).astype("float32")

complex_model = MLPClassifier(hidden_layer_sizes=(192,96,48), activation="relu", solver="adam", alpha=0.0005, batch_size=64, learning_rate_init=0.001, max_iter=500, early_stopping=True, validation_fraction=0.15, n_iter_no_change=25, random_state=SEED)
complex_model.fit(Xq_train_t, yq_train)
probs = complex_model.predict_proba(Xq_test_t)
pred_cls = probs.argmax(axis=1)
complex_acc = float(accuracy_score(yq_test, pred_cls))
joblib.dump(complex_model, MODEL_DIR / "complexity_classifier.joblib")
joblib.dump(pre_q, MODEL_DIR / "complexity_preprocessor.joblib")
joblib.dump(labeler, MODEL_DIR / "complexity_label_encoder.joblib")

# Metadata used by the app.
metadata = {
    "dataset_rows": int(len(df)),
    "project_types": sorted(df["Project_Type"].unique().tolist()),
    "project_names": sorted(df["Project_Name"].unique().tolist()),
    "feasibility_labels": sorted(df["Feasibility_Label"].unique().tolist()),
    "scope_complexities": sorted(df["Scope_Complexity"].unique().tolist()),
    "time_min": int(df["Time_Estimate_Days"].min()),
    "time_max": int(df["Time_Estimate_Days"].max()),
    "cost_metrics": {"mae_usd": cost_mae, "rmse_usd": cost_rmse, "r2": cost_r2},
    "complexity_accuracy": complex_acc,
    "cost_features": COST_FEATURES,
    "complexity_features": COMPLEX_FEATURES,
    "complexity_classes": labeler.classes_.tolist()
}
(MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(json.dumps(metadata, indent=2))
