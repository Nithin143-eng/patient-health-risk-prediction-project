"""Train the educational Random Forest model using the included synthetic dataset."""
import json
from pathlib import Path
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT=Path(__file__).parent
df=pd.read_csv(ROOT/"data"/"synthetic_patient_data.csv")
features=["age","gender","bmi","systolic_bp","diastolic_bp","glucose","cholesterol","heart_rate","smoking","physical_activity","family_history"]
numeric=["age","bmi","systolic_bp","diastolic_bp","glucose","cholesterol","heart_rate","smoking","family_history"]
categorical=["gender","physical_activity"]
pre=ColumnTransformer([
 ("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())]),numeric),
 ("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),categorical)
])
pipeline=Pipeline([("preprocess",pre),("classifier",RandomForestClassifier(n_estimators=220,max_depth=12,min_samples_leaf=3,class_weight="balanced",random_state=42))])
X_train,X_test,y_train,y_test=train_test_split(df[features],df["risk_level"],test_size=.2,random_state=42,stratify=df["risk_level"])
pipeline.fit(X_train,y_train)
pred=pipeline.predict(X_test)
metrics={"accuracy":float(accuracy_score(y_test,pred)),"train_size":len(y_train),"test_size":len(y_test),
 "labels":["Low","Moderate","High"],"confusion_matrix":confusion_matrix(y_test,pred,labels=["Low","Moderate","High"]).tolist(),
 "classification_report":classification_report(y_test,pred,output_dict=True,zero_division=0)}
(ROOT/"models").mkdir(exist_ok=True)
joblib.dump(pipeline,ROOT/"models"/"risk_model.joblib")
(ROOT/"models"/"metrics.json").write_text(json.dumps(metrics,indent=2))
print(f"Model saved. Test accuracy: {metrics['accuracy']:.3f}")
