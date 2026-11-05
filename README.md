# Patient Health Risk Prediction using Data Analytics

An academic demonstration application that explores synthetic patient-like health data, trains a Random Forest classifier, provides interactive predictions, and stores prediction history locally.

## Important safety note
This project uses **synthetic, artificially generated data and labels**. It is not a validated clinical model, medical device, diagnosis, or treatment tool. Do not use it for real clinical decisions and do not enter identifiable patient information.

## Features
- Streamlit dashboard with summary metrics and charts
- Synthetic dataset with 2,400 records
- Dataset explorer, filters, descriptive statistics, CSV export
- Data preprocessing with scikit-learn pipelines
- Random Forest multi-class classification (Low / Moderate / High)
- Held-out test evaluation, confusion matrix and classification report
- Interactive prediction form with class probability visualization
- SQLite prediction history and CSV export
- Streamlit Cloud deployment configuration

## Project structure
```
patient_health_risk_prediction/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── synthetic_patient_data.csv
├── models/
│   ├── risk_model.joblib
│   └── metrics.json
├── notebooks/
│   └── Patient_Health_Risk_Analysis.ipynb
├── tests/
│   └── test_dataset.py
└── .streamlit/
    └── config.toml
```

## Run in Google Colab (no local installation)
1. Upload this ZIP to Google Drive and extract it, or upload the project folder to a GitHub repository.
2. Open a new Colab notebook.
3. Upload `data/synthetic_patient_data.csv` and `train_model.py` plus the project files, or clone your GitHub repository in Colab.
4. In a Colab cell, install dependencies:
   `!pip install -r requirements.txt`
5. Train the model:
   `!python train_model.py`
6. Download the generated `models/risk_model.joblib` if you are updating the model artifact.

Colab free resources are subject to availability and usage limits.

## Run locally (optional)
From the project root:
```
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
Keep the terminal open while using the app.

## Deploy free using Streamlit Community Cloud
1. Create a GitHub repository and upload all project files (keep `app.py`, `requirements.txt`, `data/`, and `models/` in the repository).
2. Visit https://share.streamlit.io/ and sign in with GitHub.
3. Select **Create app**.
4. Select your repository, branch `main`, and main file path `app.py`.
5. Click **Deploy**. Streamlit installs packages from `requirements.txt`.
6. Check the app logs if the deployment fails.

The app's SQLite history is local to the app runtime. Do not assume it is permanent across restarts/redeployments. For durable multi-user storage, configure a proper database service and secrets.

## Model methodology
- Features: age, gender, BMI, systolic/diastolic blood pressure, glucose, cholesterol, heart rate, smoking, physical activity, family history.
- Numeric preprocessing: median imputation and standard scaling.
- Categorical preprocessing: most-frequent imputation and one-hot encoding.
- Classifier: RandomForestClassifier.
- Split: stratified 80/20 train/test.
- Reproducibility: fixed random seeds.

## Dataset
`data/synthetic_patient_data.csv` contains generated sample values. The target label was assigned using a synthetic scoring rule, not a medical guideline. Consequently, reported test metrics only measure how well the model learns that artificial rule.
