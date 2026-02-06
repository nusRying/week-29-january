"""Create binary (benign vs malignant) metadata for ISIC2019."""

from __future__ import annotations

import argparse
import pandas as pd
from pathlib import Path

BENIGN = {"NV", "BKL", "DF", "VASC"}
MALIGNANT = {"MEL", "BCC", "SCC"}
TARGET_CLASSES = BENIGN | MALIGNANT
TARGET_LIST = sorted(TARGET_CLASSES)


def load_ground_truth(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"image"} | TARGET_CLASSES
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    return df[["image", *TARGET_LIST]]


def build_binary_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["primary_label"] = df[TARGET_LIST].idxmax(axis=1)
    df["max_prob"] = df[TARGET_LIST].max(axis=1)
    df = df[df["primary_label"].notna()]
    df = df[df["primary_label"].isin(TARGET_CLASSES)]
    df["binary_label"] = df["primary_label"].apply(lambda x: 0 if x in BENIGN else 1)
    df = df[["image", "primary_label", "binary_label", "max_prob"]]
    return df.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("ISIC_2019_Training_GroundTruth.csv"),
        help="Path to ISIC2019 ground truth CSV",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("binary_experiment_metadata.csv"),
        help="Where to write the filtered binary metadata",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_ground_truth(args.csv)
    binary_df = build_binary_metadata(df)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    binary_df.to_csv(args.out, index=False)
    print(f"Wrote {len(binary_df)} records to {args.out}")


if __name__ == "__main__":
    main()
