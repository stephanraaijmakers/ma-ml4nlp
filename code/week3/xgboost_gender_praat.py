#!/usr/bin/env python3
import argparse
import sys
import os
import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from joblib import dump
from xgboost import XGBClassifier

FIXED_COLUMNS = [
    "file",
    "duration_s",
    "voiced_ratio",
    "f0_mean",
    "f0_median",
    "f0_min",
    "f0_max",
    "f0_std",
    "jitter_local",
    "shimmer_local",
    "hnr_mean_db",
    "intensity_mean_db",
    "intensity_std_db",
    "formant1_mean_hz",
    "formant2_mean_hz",
    "formant3_mean_hz",
    "label"
]

def parse_args():
    p = argparse.ArgumentParser(description="Train XGBoost classifier on male/female dataset (header ignored).")
    p.add_argument("--csv", required=True, help="Path to male_female_1000.csv (header will be ignored).")
    p.add_argument("--test_size", type=float, default=0.20, help="Test split fraction (default 0.20)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--cv", type=int, default=0, help="If >0, run StratifiedKFold cross-validation")
    p.add_argument("--importance", action="store_true", help="Print top 25 feature importances")
    p.add_argument("--model_out", default="model_xgb.pkl", help="File to save trained model bundle")
    p.add_argument("-i", "--ignore",
                   help="Semicolon/comma-separated 1-based indices of feature columns to ignore "
                        "(after removing 'file' and 'label'). Example: --ignore 1;4;6")
    p.add_argument("--n-estimators", type=int, default=300, help="Number of boosting trees (default 300)")
    p.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate / eta (default 0.05)")
    p.add_argument("--max-depth", type=int, default=5, help="Maximum tree depth (default 5)")
    return p.parse_args()

def parse_ignore_indices(spec: str, max_index: int):
    raw = re.split(r'[;,]+', spec.strip())
    idxs = []
    for r in raw:
        if not r:
            continue
        try:
            v = int(r)
            if v < 1 or v > max_index:
                print(f"[WARN] Ignoring out-of-range feature index {v} (valid 1..{max_index})")
            else:
                idxs.append(v)
        except ValueError:
            print(f"[WARN] Could not parse ignore index '{r}'")
    return sorted(set(idxs))

def main():
    args = parse_args()

    if not os.path.isfile(args.csv):
        print(f"File not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading (ignoring header) from {args.csv} ...")
    df = pd.read_csv(args.csv, header=None)
    if df.shape[1] != len(FIXED_COLUMNS):
        print(f"Unexpected number of columns: got {df.shape[1]}, expected {len(FIXED_COLUMNS)}", file=sys.stderr)
        sys.exit(2)
    df.columns = FIXED_COLUMNS

    # Clean label
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(["male", "female"])]
    if df.empty:
        print("No valid male/female labels after cleaning.", file=sys.stderr)
        sys.exit(3)

    print(f"Rows after label filtering: {len(df)}")

    # Features (drop file + label)
    feature_cols = [c for c in df.columns if c not in ("file", "label")]

    # Handle ignore indices
    if args.ignore:
        max_idx = len(feature_cols)
        ignore_indices = parse_ignore_indices(args.ignore, max_idx)  # 1-based indices
        if ignore_indices:
            to_drop = [feature_cols[i - 1] for i in ignore_indices]
            feature_cols = [c for c in feature_cols if c not in to_drop]
            print(f"Ignoring feature indices {ignore_indices} -> columns {to_drop}")
        else:
            print("[INFO] No valid feature indices to ignore parsed.")

    if not feature_cols:
        print("All features removed after applying ignore list.", file=sys.stderr)
        sys.exit(4)

    # Numeric conversion
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        X = X.fillna(X.mean())

    y_raw = df["label"]
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Report class distribution
    uniques, counts = np.unique(y, return_counts=True)
    print("Class distribution:")
    for cls_idx, cnt in zip(uniques, counts):
        print(f"  {le.inverse_transform([cls_idx])[0]}: {cnt}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    model = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=args.seed,
        n_jobs=-1
    )

    print("Training...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    if args.cv and args.cv > 1:
        print(f"\n{args.cv}-fold CV (accuracy)...")
        cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        print(f"CV mean={scores.mean():.4f} std={scores.std():.4f} scores={scores}")

    if args.importance:
        booster = model.get_booster()
        gains = booster.get_score(importance_type="gain")
        fmap = {f"f{i}": col for i, col in enumerate(feature_cols)}
        ranked = sorted(((fmap.get(k, k), v) for k, v in gains.items()), key=lambda x: x[1], reverse=True)[:25]
        print("\nTop feature importances (gain):")
        for name, val in ranked:
            print(f"  {name}: {val:.6f}")

    dump({"model": model, "label_encoder": le, "feature_columns": feature_cols}, args.model_out)
    print(f"\nSaved model bundle to {args.model_out}")

    print("\nReload example:")
    print(f"from joblib import load; b=load('{args.model_out}'); model=b['model']")

if __name__ == "__main__":
    main()