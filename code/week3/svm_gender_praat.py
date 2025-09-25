#!/usr/bin/env python3
import argparse
import sys
import os
import re
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from joblib import dump

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
    p = argparse.ArgumentParser(
        description="Train an SVM classifier on male/female Praat features (header ignored).",
        allow_abbrev=False
    )
    p.add_argument("--csv", required=True, help="Path to male_female_1000.csv (header will be ignored).")
    p.add_argument("--test_size", type=float, default=0.20, help="Test split fraction")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--cv", type=int, default=0, help="If >0, run StratifiedKFold cross-validation")

    # Feature control
    p.add_argument("-i", "--ignore", help="Semicolon/comma-separated 1-based indices to ignore after removing 'file' and 'label' (e.g., -i 1;4;6)")
    p.add_argument("--no-scale", action="store_true", help="Disable StandardScaler (enabled by default)")

    # SVM hyperparameters
    p.add_argument("--kernel", choices=["rbf", "linear", "poly", "sigmoid"], default="rbf", help="SVM kernel")
    p.add_argument("--C", type=float, default=1.0, help="Regularization strength (C)")
    p.add_argument("--gamma", default="scale", help="Kernel coefficient: 'scale', 'auto', or a float")
    p.add_argument("--degree", type=int, default=3, help="Degree for poly kernel")
    p.add_argument("--coef0", type=float, default=0.0, help="coef0 for poly/sigmoid")
    p.add_argument("--class-weight", dest="class_weight", choices=["none", "balanced"], default="none", help="Class weighting")
    p.add_argument("--probability", action="store_true", help="Enable probability estimates (slower)")

    # Misc
    p.add_argument("--cache-size", type=float, default=200.0, help="SVM kernel cache size (MB)")
    p.add_argument("--max-iter", type=int, default=-1, help="Hard limit on iterations (-1 = no limit)")
    p.add_argument("--model_out", default="model_svm.pkl", help="Path to save trained model bundle")

    # Feature importance
    p.add_argument("--importance", action="store_true", help="Compute and print feature importance")
    p.add_argument("--pi-repeats", type=int, default=10, help="Permutation importance repeats (non-linear kernels)")
    p.add_argument("--pi-n-jobs", type=int, default=-1, help="Permutation importance parallel jobs")
    return p.parse_args()

def parse_ignore_indices(spec: str, max_index: int):
    raw = re.split(r"[;,]+", spec.strip())
    idxs = []
    for r in raw:
        if not r:
            continue
        try:
            v = int(r)
            if 1 <= v <= max_index:
                idxs.append(v)
            else:
                print(f"[WARN] Ignoring out-of-range feature index {v} (valid 1..{max_index})")
        except ValueError:
            print(f"[WARN] Could not parse ignore index '{r}'")
    return sorted(set(idxs))

def parse_gamma_value(gamma_str: str):
    s = str(gamma_str).strip().lower()
    if s in {"scale", "auto"}:
        return s
    try:
        return float(s)
    except ValueError:
        raise ValueError("--gamma must be 'scale', 'auto', or a float")

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

    # Clean label, keep only male/female
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(["male", "female"])]
    if df.empty:
        print("No valid male/female labels after cleaning.", file=sys.stderr)
        sys.exit(3)

    # Build feature list (drop file + label)
    feature_cols = [c for c in df.columns if c not in ("file", "label")]

    # Apply ignore list
    if args.ignore:
        max_idx = len(feature_cols)
        ignore_indices = parse_ignore_indices(args.ignore, max_idx)
        if ignore_indices:
            to_drop = [feature_cols[i - 1] for i in ignore_indices]  # 1-based -> 0-based
            feature_cols = [c for c in feature_cols if c not in to_drop]
            print(f"Ignoring feature indices {ignore_indices} -> columns {to_drop}")
        else:
            print("[INFO] No valid feature indices parsed for --ignore")

    if not feature_cols:
        print("All features removed after applying ignore list.", file=sys.stderr)
        sys.exit(4)

    # Convert to numeric and impute
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        X = X.fillna(X.mean())

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df["label"])

    # Report class distribution
    uniques, counts = np.unique(y, return_counts=True)
    print("Class distribution:")
    for cls_idx, cnt in zip(uniques, counts):
        print(f"  {le.inverse_transform([cls_idx])[0]}: {cnt}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # Build pipeline: scaler (optional) + SVC
    steps = []
    scaler = None
    if not args.no_scale:
        scaler = StandardScaler()
        steps.append(("scaler", scaler))
    svc = SVC(
        kernel=args.kernel,
        C=args.C,
        gamma=parse_gamma_value(args.gamma),
        degree=args.degree,
        coef0=args.coef0,
        class_weight=(None if args.class_weight == "none" else "balanced"),
        probability=args.probability,
        cache_size=args.cache_size,
        max_iter=args.max_iter,
    )
    steps.append(("svc", svc))
    model = Pipeline(steps)

    print("Training SVM...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Optional ROC AUC if binary
    try:
        if args.probability:
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)
        auc = roc_auc_score(y_test, y_score)
        print(f"ROC AUC: {auc:.4f}")
    except Exception:
        pass

    # Cross-validation (on full data)
    if args.cv and args.cv > 1:
        print(f"\n{args.cv}-fold CV (accuracy)...")
        skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
        cv_scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
        print(f"CV mean={cv_scores.mean():.4f} std={cv_scores.std():.4f} scores={np.array2string(cv_scores, precision=4)}")

    # Feature importance
    if args.importance:
        print("\nFeature importance:")
        printed_any = False

        # Linear SVM: coefficients as importances (on standardized features if scaling enabled)
        try:
            svc_fitted = model.named_steps["svc"]
            if args.kernel == "linear" and hasattr(svc_fitted, "coef_"):
                coefs = svc_fitted.coef_.ravel()
                importances = np.abs(coefs)
                ranked_idx = np.argsort(importances)[::-1]
                print("Linear SVM coefficients (absolute, on standardized features if scaling enabled):")
                for i in ranked_idx[:25]:
                    print(f"  {feature_cols[i]}: {importances[i]:.6f}")
                printed_any = True
        except Exception:
            pass

        # Permutation importance for any kernel
        try:
            from sklearn.inspection import permutation_importance
            r = permutation_importance(
                model, X_test, y_test,
                n_repeats=args.pi_repeats,
                n_jobs=args.pi_n_jobs,
                random_state=args.seed,
                scoring="accuracy"
            )
            means = r.importances_mean
            stds = r.importances_std
            ranked_idx = np.argsort(means)[::-1]
            print("\nPermutation importance (decrease in accuracy):")
            for i in ranked_idx[:25]:
                print(f"  {feature_cols[i]}: mean={means[i]:.6f} std={stds[i]:.6f}")
            printed_any = True
        except Exception as e:
            if not printed_any:
                print(f"[WARN] Could not compute feature importance: {e}")

    # Save bundle
    dump({
        "model": model,
        "label_encoder": le,
        "feature_columns": feature_cols,
        "params": {
            "kernel": args.kernel,
            "C": args.C,
            "gamma": args.gamma,
            "degree": args.degree,
            "coef0": args.coef0,
            "scaled": not args.no_scale,
        }
    }, args.model_out)
    print(f"\nSaved model bundle to {args.model_out}")
    print(f"Reload: from joblib import load; b = load('{args.model_out}'); clf = b['model']")

if __name__ == "__main__":
    main()