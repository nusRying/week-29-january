"""Train ExSTraCS binary classifier; auto-runs feature extraction when needed."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("binary_features.csv"),
        help="Path to features CSV",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("binary_experiment_metadata.csv"),
        help="Path to metadata CSV (used if features missing)",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("ISIC_2019_Training_Input"),
        help="Images dir (passed to extractor)",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("binary_exstracs_model.pkl"),
        help="Where to save the trained model",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def ensure_features(args: argparse.Namespace) -> None:
    if args.features.exists():
        return
    cmd = [
        sys.executable,
        "extract_binary_features.py",
        "--metadata",
        str(args.metadata),
        "--images",
        str(args.images),
        "--out",
        str(args.features),
    ]
    print(f"[INFO] Features missing; running extractor: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def load_exstracs() -> object:
    root = Path(__file__).resolve().parent.parent / "scikit-ExSTraCS-master"
    sys.path.append(str(root))
    try:
        from skExSTraCS import ExSTraCS  # type: ignore
    except ImportError as exc:  # noqa: BLE001
        raise ImportError(
            "Could not import ExSTraCS. Ensure scikit-ExSTraCS-master is present."
        ) from exc
    return ExSTraCS


def main() -> None:
    args = parse_args()
    ensure_features(args)

    df = pd.read_csv(args.features)
    feature_cols = [c for c in df.columns if c not in {"image", "primary_label", "binary_label"}]
    X = df[feature_cols].values
    y = df["binary_label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )

    ros = RandomOverSampler(random_state=args.seed)
    X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_res = scaler.fit_transform(X_train_res)
    X_test = scaler.transform(X_test)

    ExSTraCS = load_exstracs()
    model = ExSTraCS(
        N=3000,
        iterations=200_000,
        nu=5,
        random_state=args.seed,
    )
    model.fit(X_train_res, y_train_res)

    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn + 1e-9)
    specificity = tn / (tn + fp + 1e-9)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler, "features": feature_cols}, args.model_out)

    print(f"Sensitivity (TPR): {sensitivity:.4f}")
    print(f"Specificity (TNR): {specificity:.4f}")
    print(f"Saved model to {args.model_out}")


if __name__ == "__main__":
    main()
