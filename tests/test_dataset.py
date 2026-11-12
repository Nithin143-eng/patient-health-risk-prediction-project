from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
def test_dataset_exists_and_schema():
    path=ROOT/"data"/"synthetic_patient_data.csv"
    assert path.exists()
    df=pd.read_csv(path)
    expected={"age","gender","bmi","systolic_bp","diastolic_bp","glucose","cholesterol","heart_rate","smoking","physical_activity","family_history","risk_level"}
    assert expected.issubset(df.columns)
    assert len(df)>=1000
    assert set(df["risk_level"].unique()) <= {"Low","Moderate","High"}
