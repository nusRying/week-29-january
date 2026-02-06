"""Extract features for the binary skin lesion experiment with resume support."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Expect local helper modules to be supplied by user.
from preprocessing_utils import apply_hair_removal as hair_removal  # type: ignore
from preprocessing_utils import apply_color_constancy as gray_world_normalization  # type: ignore
from feature_engine import extract_feature_vector  # type: ignore


def load_metadata(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"image", "primary_label", "binary_label"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Metadata missing columns: {missing}")
    return df


def already_processed(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    df_done = pd.read_csv(out_path)
    return set(df_done["image"].astype(str))


def process_image(image_path: Path) -> Dict:
    import cv2  # local import to keep base import light
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image {image_path}")
    img = hair_removal(img)
    img = gray_world_normalization(img)
    feats = extract_feature_vector(img)
    return feats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("binary_experiment_metadata.csv"),
        help="Input metadata with image + binary_label",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("ISIC_2019_Training_Input"),
        help="Directory containing training images",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("binary_features.csv"),
        help="Output CSV with features",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Save progress every N images",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta = load_metadata(args.metadata)
    processed = already_processed(args.out)

    rows: List[Dict] = []
    total = len(meta)
    for idx, row in meta.iterrows():
        image_id = str(row["image"])
        if image_id in processed:
            continue

        image_path = args.images / f"{image_id}.jpg"
        try:
            feats = process_image(image_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Skipping {image_id}: {exc}")
            continue

        record = {
            "image": image_id,
            "primary_label": row["primary_label"],
            "binary_label": row["binary_label"],
            **feats,
        }
        rows.append(record)
        processed.add(image_id)

        if len(rows) >= args.save_every:
            _append_rows(args.out, rows)
            rows = []
            print(f"[PROGRESS] {idx + 1}/{total} processed; saved checkpoint.")

    if rows:
        _append_rows(args.out, rows)
        print(f"[PROGRESS] Final save with {len(rows)} rows.")

    print(f"Finished. Features in {args.out}")


def _append_rows(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()
