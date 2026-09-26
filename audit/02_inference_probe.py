"""
Audit step 2 — inspect the SHIPPED model bundle and probe the inference paths
(FastAPI app and CLI) for behavioural defects.

Run from the repo root:
    python audit/02_inference_probe.py
Requires: requirements.txt + httpx (for FastAPI's TestClient).
"""
import contextlib
import io
import subprocess
import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from preprocessing import preprocess_inference  # noqa: E402

# ── 1. Bundle internals ──────────────────────────────────────────────────────
bundle = joblib.load(REPO / "models" / "best_model_bundle.joblib")
model = bundle["model"]
cols = bundle["feature_cols"]
print(f"model: {bundle['model_name']} ({type(model).__name__})  threshold: {bundle['threshold']}")
sel = bundle["stat_params"]["selector"]
print(f"dropped (low variance): {sel['low_var_cols']}   dropped (high corr): {sel['high_corr_cols']}")

print("\nIQR fences fitted on training data — (0, 0) means the column is clipped to all-zeros:")
for k, (lo, hi) in bundle["stat_params"]["fences"].items():
    flag = "  <-- constant 0" if lo == 0 and hi == 0 else ""
    print(f"  {k:18s} [{lo:8.3f}, {hi:8.3f}]{flag}")

print("\nbase LR coefficients (0 => feature has no effect):")
print(pd.Series(model.base_model.coef_[0], index=cols).round(4).to_string())

# ── 2. FastAPI behavioural probe ─────────────────────────────────────────────
from fastapi.testclient import TestClient  # noqa: E402
import app_api  # noqa: E402

client = TestClient(app_api.app, raise_server_exceptions=False)
base = dict(age=55, sex=1, is_smoking=0, cigsPerDay=0, BPMeds=0, prevalentStroke=0,
            prevalentHyp=0, diabetes=0, totChol=220, sysBP=130, diaBP=85, BMI=26,
            heartRate=75, glucose=90, education=2)


def api(**kw):
    return client.post("/predict", json={**base, **kw}).json()["chd_probability"]


print("\nAPI /predict — predicted 10-year CHD probability:")
for name, kw in [("baseline", {}),
                 ("diabetes=1, glucose=250", dict(diabetes=1, glucose=250)),
                 ("glucose=300", dict(glucose=300)),
                 ("BPMeds=1", dict(BPMeds=1)),
                 ("prevalentStroke=1", dict(prevalentStroke=1)),
                 ("BMI=35 (obese)", dict(BMI=35)),
                 ("BP 180/135 (same pulse pressure)", dict(sysBP=180, diaBP=135)),
                 ("is_smoking=1, cigsPerDay=0", dict(is_smoking=1)),
                 ("is_smoking=0, cigsPerDay=30 (inconsistent, accepted)", dict(cigsPerDay=30)),
                 ("age=20 (outside training range 32-70)", dict(age=20)),
                 ("age=100 (outside training range 32-70)", dict(age=100))]:
    print(f"  {name:52s} {api(**kw):.4f}")

r = client.post("/predict/batch", json=[])
print(f"\nAPI /predict/batch with [] -> HTTP {r.status_code}")

# ── 3. Subgroup calibration of the shipped model (whole dataset) ─────────────
df = pd.read_csv(REPO / "data_cardiovascular_risk.csv").drop(columns=["id"])
art = {k: bundle[k] for k in ("fill_values", "stat_params", "scaler", "feature_cols")}
with contextlib.redirect_stdout(io.StringIO()):
    df["p"] = model.predict_proba(preprocess_inference(df.drop(columns="TenYearCHD"), art))[:, 1]
groups = {"all": df.index == df.index, "diabetes=1": df.diabetes == 1,
          "prevalentStroke=1": df.prevalentStroke == 1, "BPMeds=1": df.BPMeds == 1,
          "glucose>=126": df.glucose >= 126, "sysBP>=160": df.sysBP >= 160}
print("\nobserved vs predicted risk by subgroup (shipped model):")
print(pd.DataFrame([{"subgroup": k, "n": int(m.sum()),
                     "observed": round(df.loc[m, "TenYearCHD"].mean(), 3),
                     "predicted": round(df.loc[m, "p"].mean(), 3)} for k, m in groups.items()])
      .to_string(index=False))
print(f"max predicted probability over all 3,390 patients: {df.p.max():.3f}")

# ── 4. CLI probe ─────────────────────────────────────────────────────────────
print("\nCLI src/predict.py:")
args = ["--age", "55", "--is_smoking", "0", "--totChol", "250", "--sysBP", "140",
        "--diaBP", "90", "--BMI", "28.5", "--heartRate", "75", "--glucose", "100"]
for sex in ["M", "1", "0"]:
    out = subprocess.run([sys.executable, "-W", "ignore", str(REPO / "src" / "predict.py"),
                          "--sex", sex, *args], capture_output=True, text=True)
    last = (out.stderr if out.returncode else out.stdout).strip().splitlines()[-1]
    print(f"  --sex {sex}: {'CRASH ' if out.returncode else ''}{last.strip()}")
csv_in = pd.DataFrame([{**base, "sex": s, "is_smoking": "NO", "totChol": 250, "sysBP": 140,
                        "diaBP": 90, "BMI": 28.5, "glucose": 100} for s in ("M", "F")])
with contextlib.redirect_stdout(io.StringIO()):
    p_csv = model.predict_proba(preprocess_inference(csv_in, art))[:, 1]
print(f"  same patient via raw 'M'/'F' labels (CSV path): M={p_csv[0]:.4f}  F={p_csv[1]:.4f}")

# ── 5. Can the bundle be loaded outside the app's sys.path hack? ─────────────
code = "import joblib; joblib.load('models/best_model_bundle.joblib'); print('loaded OK')"
out = subprocess.run([sys.executable, "-W", "ignore", "-c", code], cwd=REPO,
                     capture_output=True, text=True)
print("\nload bundle from repo root without src/ on sys.path ->",
      (out.stdout.strip() or out.stderr.strip().splitlines()[-1]))
