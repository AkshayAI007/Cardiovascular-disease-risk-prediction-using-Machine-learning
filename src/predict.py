"""
Inference utility — single patient or batch CSV.

Usage (single patient):
    python src/predict.py --age 55 --sex M --is_smoking 0 --cigsPerDay 0 \
        --BPMeds 0 --prevalentStroke 0 --prevalentHyp 1 --diabetes 0 \
        --totChol 250 --sysBP 140 --diaBP 90 --BMI 28.5 \
        --heartRate 75 --glucose 100 --education 2

Usage (batch CSV):
    python src/predict.py --csv path/to/new_patients.csv
"""
import os
import sys
import argparse
import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from config import MODEL_DIR
from preprocessing import preprocess_inference  # uses serialised artifacts only


BUNDLE_PATH = os.path.join(MODEL_DIR, "best_model_bundle.joblib")


def load_bundle(path: str = BUNDLE_PATH) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model bundle not found at {path}. Run train.py first.")
    return joblib.load(path)


def predict(df: pd.DataFrame, bundle: dict,
            threshold: float | None = None) -> pd.DataFrame:
    """
    Apply the full inference preprocessing pipeline (using serialised
    training statistics — never re-fit on inference data) then predict.
    """
    artifacts = {
        "fill_values":  bundle["fill_values"],
        "stat_params":  bundle["stat_params"],
        "scaler":       bundle["scaler"],
        "feature_cols": bundle["feature_cols"],
    }
    X = preprocess_inference(df, artifacts)

    model = bundle["model"]
    # Use the threshold that was tuned on the validation set during training
    t = threshold if threshold is not None else bundle.get("threshold", 0.5)

    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= t).astype(int)

    result = df.copy()
    result["chd_probability"] = probs.round(4)
    result["chd_prediction"]  = preds
    result["risk_label"]      = result["chd_prediction"].map(
        {0: "Low Risk", 1: "High Risk"})
    return result


def build_single_patient_df(args) -> pd.DataFrame:
    sex_map = {"m": 1, "f": 0, "male": 1, "female": 0}
    sex_val = sex_map.get(str(args.sex).lower(), int(args.sex))
    return pd.DataFrame([{
        "age":             float(args.age),
        "sex":             sex_val,
        "is_smoking":      int(args.is_smoking),
        "cigsPerDay":      float(args.cigsPerDay),
        "BPMeds":          int(args.BPMeds),
        "prevalentStroke": int(args.prevalentStroke),
        "prevalentHyp":    int(args.prevalentHyp),
        "diabetes":        int(args.diabetes),
        "totChol":         float(args.totChol),
        "sysBP":           float(args.sysBP),
        "diaBP":           float(args.diaBP),
        "BMI":             float(args.BMI),
        "heartRate":       float(args.heartRate),
        "glucose":         float(args.glucose),
        "education":       float(args.education),
    }])


def main() -> None:
    parser = argparse.ArgumentParser(description="Cardiovascular Risk Predictor")
    parser.add_argument("--csv",       type=str,   default=None)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override the val-tuned threshold from training")
    parser.add_argument("--age",             type=float)
    parser.add_argument("--sex")
    parser.add_argument("--is_smoking",      type=int)
    parser.add_argument("--cigsPerDay",      type=float, default=0)
    parser.add_argument("--BPMeds",          type=int,   default=0)
    parser.add_argument("--prevalentStroke", type=int,   default=0)
    parser.add_argument("--prevalentHyp",    type=int,   default=0)
    parser.add_argument("--diabetes",        type=int,   default=0)
    parser.add_argument("--totChol",         type=float, default=200)
    parser.add_argument("--sysBP",           type=float, default=120)
    parser.add_argument("--diaBP",           type=float, default=80)
    parser.add_argument("--BMI",             type=float, default=25)
    parser.add_argument("--heartRate",       type=float, default=75)
    parser.add_argument("--glucose",         type=float, default=85)
    parser.add_argument("--education",       type=float, default=2)
    args = parser.parse_args()

    bundle = load_bundle()
    print(f"Loaded: {bundle['model_name']}  "
          f"(val-tuned threshold: {bundle.get('threshold', 0.5):.2f})")

    df = pd.read_csv(args.csv) if args.csv else build_single_patient_df(args)

    results = predict(df, bundle, threshold=args.threshold)
    print("\n--- Prediction Results ---")
    print(results[["chd_probability", "chd_prediction", "risk_label"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
