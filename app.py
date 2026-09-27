from pathlib import Path
import json, joblib, numpy as np, pandas as pd
from flask import Flask, render_template, request, jsonify

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "data" / "Engineering_Cost_Feasibility_Dataset.csv"
MODEL_DIR = BASE / "models"

df = pd.read_csv(DATA_PATH)
metadata = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
cost_model = joblib.load(MODEL_DIR / "cost_predictor.joblib")
complex_model = joblib.load(MODEL_DIR / "complexity_classifier.joblib")
cost_pre = joblib.load(MODEL_DIR / "cost_preprocessor.joblib")
complex_pre = joblib.load(MODEL_DIR / "complexity_preprocessor.joblib")
labeler = joblib.load(MODEL_DIR / "complexity_label_encoder.joblib")

app = Flask(__name__)

TYPE_IMAGES = {
    "Road": "road.svg",
    "Building": "building.svg",
    "Bridge": "bridge.svg",
    "Power Plant": "power.svg",
    "Water Infra": "water.svg",
}
TYPE_DESCRIPTIONS = {
    "Road": "Road corridors, urban roads, highways and transport upgrades.",
    "Building": "Residential, commercial and institutional building projects.",
    "Bridge": "Bridge construction, rehabilitation and structural infrastructure.",
    "Power Plant": "Power generation and energy infrastructure projects.",
    "Water Infra": "Water supply, treatment, drainage and utility infrastructure.",
}

def nearest_reference(project_type, project_name, feasibility, time_days):
    pool = df[(df.Project_Type == project_type) & (df.Project_Name == project_name) & (df.Feasibility_Label == feasibility)].copy()
    level = "Exact project profile"
    if pool.empty:
        pool = df[(df.Project_Type == project_type) & (df.Project_Name == project_name)].copy()
        level = "Project profile"
    if pool.empty:
        pool = df[df.Project_Type == project_type].copy()
        level = "Project type"
    if pool.empty:
        pool = df.copy()
        level = "Dataset average"
    pool["_distance"] = (pool["Time_Estimate_Days"] - float(time_days)).abs()
    return pool.sort_values("_distance").iloc[0], level

def build_inputs(row, overrides):
    vals = row.copy()
    for k,v in overrides.items():
        if v is not None and v != "": vals[k] = v
    return vals

def to_record(row):
    out = row.drop(labels=[c for c in ["_distance"] if c in row.index]).to_dict()
    for k,v in list(out.items()):
        if isinstance(v, (np.integer,)): out[k]=int(v)
        elif isinstance(v, (np.floating,)): out[k]=float(v)
    return out

@app.route("/")
def index():
    return render_template("index.html", metadata=metadata, type_images=TYPE_IMAGES, type_descriptions=TYPE_DESCRIPTIONS)

@app.get("/api/project-names")
def project_names():
    t = request.args.get("project_type", "")
    names = sorted(df.loc[df.Project_Type == t, "Project_Name"].unique().tolist()) if t else metadata["project_names"]
    return jsonify({"project_names": names})

@app.get("/api/project-preview")
def project_preview():
    t=request.args.get("project_type","")
    n=request.args.get("project_name","")
    pool=df[(df.Project_Type==t)&(df.Project_Name==n)] if t and n else pd.DataFrame()
    if pool.empty: return jsonify({"found":False})
    row=pool.iloc[0]
    return jsonify({"found":True,"project":to_record(row),"image":TYPE_IMAGES.get(t,"building.svg")})

@app.post("/api/predict")
def predict():
    data=request.get_json(force=True) or {}
    required=["project_type","project_name","time_days","feasibility"]
    missing=[x for x in required if data.get(x) in [None,""]]
    if missing: return jsonify({"error":"Missing fields: " + ", ".join(missing)}),400
    try: time_days=float(data["time_days"])
    except: return jsonify({"error":"Time must be a number."}),400

    ref, match_level = nearest_reference(data["project_type"], data["project_name"], data["feasibility"], time_days)
    overrides={
        "Project_Type": data["project_type"],
        "Project_Name": data["project_name"],
        "Feasibility_Label": data["feasibility"],
        "Time_Estimate_Days": time_days,
        "Scope_Complexity": data.get("scope_complexity", ref["Scope_Complexity"]),
        "Resource_Allocation_Score": float(data.get("resource_score", ref["Resource_Allocation_Score"])),
        "Risk_Assessment_Score": float(data.get("risk_score", ref["Risk_Assessment_Score"])),
        "Environmental_Impact_Score": float(data.get("environment_score", ref["Environmental_Impact_Score"])),
        "Historical_Cost_Deviation_%": float(data.get("deviation", ref["Historical_Cost_Deviation_%"])),
        "Stakeholder_Priority_Score": float(data.get("stakeholder_score", ref["Stakeholder_Priority_Score"])),
    }
    row=build_inputs(ref, overrides)
    cost_cols=metadata["cost_features"]
    complex_cols=metadata["complexity_features"]
    cost_df=pd.DataFrame([{c:row[c] for c in cost_cols}])
    complex_df=pd.DataFrame([{c:row[c] for c in complex_cols}])
    cost_x=cost_pre.transform(cost_df).astype("float32")
    complex_x=complex_pre.transform(complex_df).astype("float32")
    predicted_cost=float(np.expm1(cost_model.predict(cost_x)[0]))
    probs=complex_model.predict_proba(complex_x).ravel()
    idx=int(np.argmax(probs)); complexity=str(labeler.inverse_transform([idx])[0]); confidence=float(probs[idx]*100)
    scope_score={"Low":1,"Medium":2,"High":3}.get(complexity,2)
    # normalized score for presentation, derived from class probabilities
    complexity_score=round(float(np.dot(probs, [1,2,3])/3*10),1)
    hist_cost=float(ref["Estimated_Cost_USD"])
    delta=predicted_cost-hist_cost
    similar=df[(df.Project_Type==data["project_type"]) & (df.Project_Name==data["project_name"])].copy()
    similar["distance"]=(similar["Time_Estimate_Days"]-time_days).abs()
    similar=similar.sort_values("distance").head(4)
    similar_records=[]
    for _,r in similar.iterrows():
        similar_records.append({"project_name":r.Project_Name,"type":r.Project_Type,"cost":float(r.Estimated_Cost_USD),"days":int(r.Time_Estimate_Days),"complexity":r.Scope_Complexity})
    type_avg=df.groupby("Project_Type")["Estimated_Cost_USD"].mean().round(0).to_dict()
    feasibility_dist=df["Feasibility_Label"].value_counts().to_dict()
    history=df[(df.Project_Type==data["project_type"]) & (df.Project_Name==data["project_name"])].sort_values("Time_Estimate_Days").head(120)
    chart_history={"labels":history["Time_Estimate_Days"].astype(int).tolist(),"costs":history["Estimated_Cost_USD"].astype(float).tolist()}
    record=to_record(ref)
    record.update({"Predicted_Cost_USD":predicted_cost,"Predicted_Complexity":complexity,"Complexity_Confidence_%":confidence})
    return jsonify({
        "predicted_cost":predicted_cost,"historical_cost":hist_cost,"difference":delta,"difference_pct":(delta/hist_cost*100 if hist_cost else 0),
        "complexity":complexity,"complexity_confidence":confidence,"complexity_score":complexity_score,"reference_match":match_level,
        "record":record,"image":TYPE_IMAGES.get(data["project_type"],"building.svg"),"description":TYPE_DESCRIPTIONS.get(data["project_type"],"Engineering project"),
        "similar_projects":similar_records,"type_average":type_avg,"feasibility_distribution":feasibility_dist,"history":chart_history,
        "model_metrics":metadata["cost_metrics"] | {"complexity_accuracy":metadata["complexity_accuracy"]}
    })

@app.get("/api/health")
def health(): return jsonify({"status":"ok","model":"deep-neural-network-mlp","rows":metadata["dataset_rows"]})

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
