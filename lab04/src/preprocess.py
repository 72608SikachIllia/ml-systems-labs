"""Step 4
"""
import json

import joblib
import numpy as np
import pandas as pd

from common import (CATEGORICAL, NUMERIC, PROCESSED_DIR, RESULTS, build_preprocessor, load_raw,
                    make_split, set_seeds)


def processed_frame(pre, X, y):
    names = [n.split("__", 1)[-1] for n in pre.named_steps["columns"].get_feature_names_out()]
    out = pd.DataFrame(pre.transform(X), columns=names, index=X.index)
    out["income_gt_50k"] = y.values
    return out


def main():
    set_seeds()
    X_tr, X_te, y_tr, y_te, info = make_split(load_raw())

    pre = build_preprocessor(derived=True).fit(X_tr)
    joblib.dump(pre, RESULTS / "preprocessor.joblib")

    ct = pre.named_steps["columns"]
    num_names = list(ct.transformers_[0][2])
    num_pipe, cat_pipe = ct.named_transformers_["num"], ct.named_transformers_["cat"]
    clip = num_pipe.named_steps["clip"]
    fitted = pd.DataFrame({
        "feature": num_names,
        "median (train)": num_pipe.named_steps["impute"].statistics_,
        "clip_low p1 (train)": clip.lower_,
        "clip_high p99 (train)": clip.upper_,
    })
    fitted.to_csv(RESULTS / "fitted_numeric_stats.csv", index=False)
    modes = dict(zip(CATEGORICAL, map(str, cat_pipe.named_steps["impute"].statistics_)))

    train_p = processed_frame(pre, X_tr, y_tr)
    test_p = processed_frame(pre, X_te, y_te)
    train_p.to_csv(PROCESSED_DIR / "train.csv", index=False, float_format="%.10g")
    test_p.to_csv(PROCESSED_DIR / "test.csv", index=False, float_format="%.10g")

    log = {
        **info,
        "train_missing_before_impute": {c: int(X_tr[c].isna().sum()) for c in X_tr.columns
                                        if X_tr[c].isna().any()},
        "test_missing_before_impute": {c: int(X_te[c].isna().sum()) for c in X_te.columns
                                       if X_te[c].isna().any()},
        "categorical_modes (train)": modes,
        "train_values_clipped": int(sum(
            ((X_tr[c] < lo) | (X_tr[c] > hi)).sum()
            for c, lo, hi in zip(NUMERIC, clip.lower_[:len(NUMERIC)], clip.upper_[:len(NUMERIC)]))),
        "processed_features": int(train_p.shape[1] - 1),
        "train_positive_rate": float(np.mean(y_tr)),
        "test_positive_rate": float(np.mean(y_te)),
    }
    (RESULTS / "cleaning_log.json").write_text(json.dumps(log, indent=2))
    print(f"Train {info['train_rows_before_dedup']} -> {info['train_rows']} rows "
          f"({info['train_duplicates_dropped']} exact duplicates dropped); test {info['test_rows']} untouched")
    print(f"Processed matrix: {log['processed_features']} features; preprocessor saved")


if __name__ == "__main__":
    main()
