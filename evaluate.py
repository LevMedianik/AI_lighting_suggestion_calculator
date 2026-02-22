import argparse
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def baseline_fixture_count(total_lumens: int, fixture_lm: int) -> int:
    total_lumens = int(total_lumens or 0)
    fixture_lm = int(fixture_lm or 1)
    return max(1, math.ceil(max(0, total_lumens) / max(1, fixture_lm)))


def round_positive_int(arr) -> np.ndarray:
    arr = np.rint(arr).astype(int)
    return np.maximum(arr, 1)


def acc_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    return float(np.mean(np.abs(y_pred - y_true) <= k))


def compute_metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "acc@1": acc_at_k(y_true, y_pred, 1),
    }


def evaluate(model_path: Path, data_path: Path) -> int:
    if not model_path.exists():
        print(f"[ERROR] Model file not found: {model_path.resolve()}")
        return 2
    if not data_path.exists():
        print(f"[ERROR] Dataset file not found: {data_path.resolve()}")
        return 2

    model = joblib.load(model_path)
    df = pd.read_csv(data_path)

    required_cols = [
        "area_m2",
        "ceiling_h",
        "required_lux",
        "fixture_lm",
        "total_lumens",
        "fixtures_count",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[ERROR] Dataset missing columns: {missing}")
        print(f"Available columns: {list(df.columns)}")
        return 2

    # Features used by API model.predict([[area_m2, ceiling_h, required_lux, fixture_lm]])
    X = df[["area_m2", "ceiling_h", "required_lux", "fixture_lm"]].copy()
    y_true = df["fixtures_count"].astype(int).to_numpy()

    # ML raw predictions
    y_pred_ml_raw = round_positive_int(model.predict(X))

    # Baseline predictions (physics)
    y_pred_baseline = np.array(
        [baseline_fixture_count(tl, flm) for tl, flm in zip(df["total_lumens"], df["fixture_lm"])],
        dtype=int,
    )

    # Final predictions EXACTLY like API:
    # fixtures_count = min(max(baseline_count, pred_ml), max(1, int(baseline_count * 1.5)))
    cap = np.maximum(1, (y_pred_baseline * 1.5).astype(int))
    y_pred_final = np.minimum(np.maximum(y_pred_baseline, y_pred_ml_raw), cap).astype(int)

    rows = [
        compute_metrics("Baseline (physics)", y_true, y_pred_baseline),
        compute_metrics("ML raw (rounded)", y_true, y_pred_ml_raw),
        compute_metrics("Final (API constraints)", y_true, y_pred_final),
    ]
    out = pd.DataFrame(rows)

    # Improvement vs baseline (positive = better)
    mae_bl = float(out.loc[out["model"] == "Baseline (physics)", "MAE"].iloc[0])
    rmse_bl = float(out.loc[out["model"] == "Baseline (physics)", "RMSE"].iloc[0])

    out["MAE_vs_baseline_%"] = np.nan
    out["RMSE_vs_baseline_%"] = np.nan
    for name in ["ML raw (rounded)", "Final (API constraints)"]:
        mask = out["model"] == name
        out.loc[mask, "MAE_vs_baseline_%"] = 100.0 * (mae_bl - out.loc[mask, "MAE"]) / max(1e-9, mae_bl)
        out.loc[mask, "RMSE_vs_baseline_%"] = 100.0 * (rmse_bl - out.loc[mask, "RMSE"]) / max(1e-9, rmse_bl)

    print("\n=== Evaluation (full dataset) ===")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Error summaries
    def summarize(name: str, err: np.ndarray):
        return (
            f"{name}: mean={err.mean():.4f} std={err.std():.4f} "
            f"p50_abs={np.percentile(np.abs(err), 50):.0f} "
            f"p90_abs={np.percentile(np.abs(err), 90):.0f}"
        )

    print("\n=== Error summary (pred - true) ===")
    print(summarize("Baseline", y_pred_baseline - y_true))
    print(summarize("ML raw ", y_pred_ml_raw - y_true))
    print(summarize("Final  ", y_pred_final - y_true))

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate fixtures model vs physics baseline.")
    parser.add_argument("--model-path", default="models/fixtures_model.pkl", help="Path to .pkl model")
    parser.add_argument("--data-path", default="data/lighting_dataset.csv", help="Path to dataset CSV")
    args = parser.parse_args()
    return evaluate(Path(args.model_path), Path(args.data_path))


if __name__ == "__main__":
    raise SystemExit(main())