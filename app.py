import json
import sqlite3
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "synthetic_patient_data.csv"
MODEL_PATH = ROOT / "models" / "risk_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
DB_PATH = ROOT / "data" / "predictions.db"

st.set_page_config(page_title="Patient Health Risk Analytics", page_icon="🩺", layout="wide")

# ---------- Theme ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.block-container {padding-top:1.5rem; padding-bottom:2rem; max-width:1450px;}
.hero {padding:1.5rem 1.8rem; border-radius:18px; background:linear-gradient(120deg,#102a43,#176b87); color:white; margin-bottom:1rem;}
.hero h1 {font-size:2.1rem; margin:0 0 .35rem 0; color:white;}
.hero p {margin:0; color:#d8edf4;}
.metric-card {padding:1rem 1.2rem; border:1px solid #e6edf2; border-radius:14px; background:#fff;}
.small-muted {color:#718096; font-size:.88rem;}
.risk-high {color:#b42318; font-weight:700;}
.risk-moderate {color:#b54708; font-weight:700;}
.risk-low {color:#027a48; font-weight:700;}
div[data-testid="stMetric"] {background:#f7fafc; border:1px solid #e6edf2; padding:14px 16px; border-radius:12px;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("Trained model is missing. Run `python train_model.py` and add models/risk_model.joblib.")
        st.stop()
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            patient_label TEXT,
            age INTEGER, bmi REAL, systolic_bp INTEGER, glucose INTEGER,
            predicted_risk TEXT, confidence REAL, probabilities TEXT
        )""")
init_db()

def save_prediction(row, label, confidence, probs):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""INSERT INTO predictions
        (created_at, patient_label, age, bmi, systolic_bp, glucose, predicted_risk, confidence, probabilities)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (datetime.now().isoformat(timespec="seconds"), row.get("patient_label","Patient"),
         int(row["age"]), float(row["bmi"]), int(row["systolic_bp"]), int(row["glucose"]),
         label, float(confidence), json.dumps(probs)))

def make_features(values):
    return pd.DataFrame([{
        "age": values["age"], "gender": values["gender"], "bmi": values["bmi"],
        "systolic_bp": values["systolic_bp"], "diastolic_bp": values["diastolic_bp"],
        "glucose": values["glucose"], "cholesterol": values["cholesterol"],
        "heart_rate": values["heart_rate"], "smoking": int(values["smoking"]),
        "physical_activity": values["physical_activity"], "family_history": int(values["family_history"])
    }])

model = load_model()
data = load_data()
metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}

st.sidebar.markdown("## 🩺 HealthRisk AI")
st.sidebar.caption("Patient Health Risk Prediction")
page = st.sidebar.radio("Navigation", ["Dashboard", "Predict Risk", "Dataset Explorer", "Model Performance", "Prediction History", "About Project"])
st.sidebar.divider()
st.sidebar.caption("Educational prototype • Synthetic data")
st.sidebar.info("Not a medical device. Never use this output for diagnosis or treatment decisions.")

if page == "Dashboard":
    st.markdown('<div class="hero"><h1>Patient Health Risk Analytics</h1><p>Explore health indicators, understand model outputs, and review prediction trends.</p></div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Synthetic records", f"{len(data):,}")
    c2.metric("Health risk classes", "3")
    c3.metric("Model accuracy", f"{metrics.get('accuracy',0):.1%}")
    c4.metric("Features", "11")
    left,right = st.columns([1.35,1])
    with left:
        st.subheader("Risk distribution in sample dataset")
        counts = data["risk_level"].value_counts().reindex(["Low","Moderate","High"]).fillna(0).reset_index()
        counts.columns=["Risk level","Patients"]
        fig=px.bar(counts,x="Risk level",y="Patients",color="Risk level",text="Patients",
                   color_discrete_map={"Low":"#12a879","Moderate":"#f5a623","High":"#df5b55"})
        fig.update_layout(showlegend=False,plot_bgcolor="white",paper_bgcolor="white",margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(fig,use_container_width=True)
    with right:
        st.subheader("Age profile")
        fig=px.histogram(data,x="age",color="risk_level",nbins=18,
                         color_discrete_map={"Low":"#12a879","Moderate":"#f5a623","High":"#df5b55"})
        fig.update_layout(plot_bgcolor="white",paper_bgcolor="white",margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(fig,use_container_width=True)
    st.subheader("Project workflow")
    st.markdown("**Patient inputs → Data validation → Preprocessing pipeline → Random Forest classifier → Risk estimate → Saved history**")
    st.caption("The included dataset is synthetically generated for demonstration and does not represent real patients.")

elif page == "Predict Risk":
    st.markdown('<div class="hero"><h1>Patient Risk Assessment</h1><p>Enter sample health indicators to obtain an educational model estimate.</p></div>', unsafe_allow_html=True)
    with st.form("prediction_form"):
        a,b,c = st.columns(3)
        with a:
            patient_label=st.text_input("Patient reference (optional)", placeholder="e.g. Demo-001")
            age=st.number_input("Age",18,100,45)
            gender=st.selectbox("Gender",["Female","Male","Other"])
            bmi=st.number_input("BMI",12.0,60.0,24.5,0.1)
        with b:
            systolic=st.number_input("Systolic blood pressure (mmHg)",70,250,120)
            diastolic=st.number_input("Diastolic blood pressure (mmHg)",40,150,80)
            glucose=st.number_input("Glucose (mg/dL)",40,500,100)
            cholesterol=st.number_input("Total cholesterol (mg/dL)",80,500,185)
        with c:
            heart_rate=st.number_input("Resting heart rate (bpm)",35,200,75)
            smoking=st.selectbox("Current smoker",["No","Yes"])=="Yes"
            activity=st.selectbox("Physical activity",["Low","Moderate","High"])
            family=st.selectbox("Family history of cardiovascular/metabolic disease",["No","Yes"])=="Yes"
        submitted=st.form_submit_button("Predict health risk",type="primary",use_container_width=True)
    if submitted:
        values={"age":age,"gender":gender,"bmi":bmi,"systolic_bp":systolic,"diastolic_bp":diastolic,
                "glucose":glucose,"cholesterol":cholesterol,"heart_rate":heart_rate,
                "smoking":smoking,"physical_activity":activity,"family_history":family}
        row=make_features(values)
        prediction=model.predict(row)[0]
        probabilities=model.predict_proba(row)[0]
        class_probs=dict(zip(model.classes_,[float(x) for x in probabilities]))
        confidence=float(max(probabilities))
        save_prediction({"patient_label":patient_label or "Patient",**values},prediction,confidence,class_probs)
        colors={"Low":"#12a879","Moderate":"#f5a623","High":"#df5b55"}
        st.divider()
        st.subheader("Model output")
        st.markdown(f"### Estimated category: <span style='color:{colors[prediction]}'>{prediction}</span>",unsafe_allow_html=True)
        st.metric("Model confidence for predicted class",f"{confidence:.1%}")
        st.caption("Confidence is the model's probability estimate, not a medical certainty.")
        prob_df=pd.DataFrame({"Risk class":list(class_probs.keys()),"Estimated probability":list(class_probs.values())})
        fig=px.bar(prob_df,x="Risk class",y="Estimated probability",color="Risk class",
                   color_discrete_map={"Low":"#12a879","Moderate":"#f5a623","High":"#df5b55"},text_auto=".1%")
        fig.update_layout(yaxis_tickformat=".0%",showlegend=False,plot_bgcolor="white")
        st.plotly_chart(fig,use_container_width=True)
        st.warning("This is an educational prediction from synthetic training data. It is not a diagnosis, clinical risk score, or substitute for a qualified healthcare professional.")

elif page == "Dataset Explorer":
    st.markdown('<div class="hero"><h1>Dataset Explorer</h1><p>Inspect the synthetic dataset used to train the demonstration model.</p></div>',unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    c1.metric("Rows",f"{data.shape[0]:,}")
    c2.metric("Columns",data.shape[1])
    c3.metric("Missing values",int(data.isna().sum().sum()))
    selected=st.multiselect("Filter risk categories",["Low","Moderate","High"],default=["Low","Moderate","High"])
    view=data[data["risk_level"].isin(selected)]
    st.dataframe(view,use_container_width=True,height=380)
    st.download_button("Download dataset CSV",data.to_csv(index=False).encode("utf-8"),"patient_health_dataset.csv","text/csv")
    st.subheader("Summary statistics")
    st.dataframe(view.describe(include="all").T,use_container_width=True)
    st.subheader("Explore a feature")
    feature=st.selectbox("Choose a numeric feature",["age","bmi","systolic_bp","diastolic_bp","glucose","cholesterol","heart_rate"])
    fig=px.box(view,x="risk_level",y=feature,color="risk_level",color_discrete_map={"Low":"#12a879","Moderate":"#f5a623","High":"#df5b55"})
    st.plotly_chart(fig,use_container_width=True)

elif page == "Model Performance":
    st.markdown('<div class="hero"><h1>Model Performance</h1><p>Evaluation on a held-out synthetic test split.</p></div>',unsafe_allow_html=True)
    st.metric("Test accuracy",f"{metrics.get('accuracy',0):.1%}")
    st.caption(f"Training records: {metrics.get('train_size','—')} • Test records: {metrics.get('test_size','—')}")
    cm=np.array(metrics.get("confusion_matrix",[[0,0,0],[0,0,0],[0,0,0]]))
    labels=metrics.get("labels",["Low","Moderate","High"])
    cm_df=pd.DataFrame(cm,index=labels,columns=labels)
    st.subheader("Confusion matrix")
    fig=px.imshow(cm_df,text_auto=True,labels=dict(x="Predicted",y="Actual",color="Count"),
                  color_continuous_scale="Blues",aspect="auto")
    st.plotly_chart(fig,use_container_width=True)
    report=metrics.get("classification_report",{})
    rows=[]
    for label in labels:
        if label in report:
            rows.append({"Class":label,"Precision":report[label]["precision"],"Recall":report[label]["recall"],"F1-score":report[label]["f1-score"],"Support":report[label]["support"]})
    st.subheader("Classification report")
    if rows: st.dataframe(pd.DataFrame(rows).set_index("Class").style.format("{:.3f}"),use_container_width=True)
    st.info("These metrics measure agreement with labels generated by the synthetic data rules. They do not establish clinical validity or real-world performance.")

elif page == "Prediction History":
    st.markdown('<div class="hero"><h1>Prediction History</h1><p>Review and export predictions saved in this app instance.</p></div>',unsafe_allow_html=True)
    with sqlite3.connect(DB_PATH) as conn:
        history=pd.read_sql_query("SELECT id, created_at, patient_label, age, bmi, systolic_bp, glucose, predicted_risk, confidence FROM predictions ORDER BY id DESC",conn)
    if history.empty:
        st.info("No predictions have been saved yet. Go to Predict Risk to create one.")
    else:
        st.metric("Saved predictions",len(history))
        st.dataframe(history,use_container_width=True)
        st.download_button("Download prediction history",history.to_csv(index=False).encode("utf-8"),"prediction_history.csv","text/csv")
        if st.button("Clear prediction history",type="secondary"):
            with sqlite3.connect(DB_PATH) as conn: conn.execute("DELETE FROM predictions")
            st.success("History cleared. Refresh the page to update the table.")
    st.caption("On free cloud hosting, local SQLite data may not be permanent across app restarts or redeployments.")

else:
    st.markdown('<div class="hero"><h1>About this Project</h1><p>Patient Health Risk Prediction using Data Analytics and Machine Learning.</p></div>',unsafe_allow_html=True)
    st.subheader("Objective")
    st.write("Demonstrate a complete data analytics workflow: data preparation, exploratory analysis, supervised learning, model evaluation, interactive predictions, and history tracking.")
    st.subheader("Machine learning approach")
    st.write("A Random Forest classifier is trained using a preprocessing pipeline. Numeric inputs are imputed and scaled; categorical inputs are imputed and one-hot encoded.")
    st.subheader("Input features")
    st.write(", ".join(["Age","Gender","BMI","Blood pressure","Glucose","Cholesterol","Heart rate","Smoking","Physical activity","Family history"]))
    st.subheader("Limitations and responsible use")
    st.error("This project uses synthetic, artificially generated data and labels. It is strictly an academic demonstration. It must not be used to diagnose, triage, treat, or make healthcare decisions. Do not enter identifiable patient information.")
    st.subheader("Project files")
    st.code("app.py\ntrain_model.py\nrequirements.txt\ndata/synthetic_patient_data.csv\nmodels/risk_model.joblib\nmodels/metrics.json\nnotebooks/Patient_Health_Risk_Analysis.ipynb")
