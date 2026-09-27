# Project Cost Prediction & Complexity Analysis

A professional multi-step Flask web application using the supplied Engineering Cost & Feasibility dataset and **deep neural networks (multi-layer perceptrons)**.

## What it predicts
1. Estimated project cost (regression neural network)
2. Scope complexity: Low / Medium / High (classification neural network)

## Workflow
Step 1: Project Type -> Step 2: Project -> Step 3: Parameters -> Step 4: Results -> Step 5: Analytics

## Dataset
3245 rows and 13 columns from `data/Engineering_Cost_Feasibility_Dataset.csv`.

## Deep-learning models
- Cost predictor: a 4-hidden-layer MLPRegressor (256→128→64→32) trained on log1p(cost).
- Complexity classifier: a 3-hidden-layer MLPClassifier (192→96→48); Scope_Complexity is the target and is excluded from its input features to avoid leakage.
- Preprocessors are saved with joblib.

## Windows / VS Code
Open this folder in VS Code, then PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_model.py
python app.py
```

Then open http://127.0.0.1:5000

If PowerShell blocks activation, run once:
`Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

## Important
The first training run can take several minutes because both neural networks are trained from the supplied dataset. The trained `.keras` model files are then loaded by Flask, so the web app does not retrain on every request.
