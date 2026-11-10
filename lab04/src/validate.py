"""Step 3
"""
import json

import numpy as np
import pandas as pd

from common import (DICTIONARY, EXPECTED_COLS, EXPECTED_ROWS, RANGES, RESULTS, SCHEMA, clean_str,
                    load_raw, missing_mask, schema_view)


def type_check(s: pd.Series, expected: str):
    """Return (actual dtype, ok, note)."""
    actual = str(s.dtype)
    if expected == "int":
        if pd.api.types.is_integer_dtype(s):
            return actual, True, ""
        if pd.api.types.is_float_dtype(s):
            vals = s.dropna()
            if np.all(np.mod(vals, 1) == 0):
                return actual, False, "stored as float but every value is a whole number"
            return actual, False, "non-integer values present"
        return actual, False, "not numeric"
    is_cat = isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(s) \
        or pd.api.types.is_string_dtype(s)
    return actual, bool(is_cat), "" if is_cat else "expected categorical/string"


def main():
    raw = load_raw()
    df = schema_view(raw)
    rows = []

    def add(check, column, violations, detail=""):
        rows.append({"check": check, "column": column, "violations": int(violations),
                     "detail": detail})

    # shape and column set
    add("row_count", "*", int(len(df) != EXPECTED_ROWS), f"{len(df)} rows (expected {EXPECTED_ROWS})")
    add("column_count", "*", int(df.shape[1] != EXPECTED_COLS),
        f"{df.shape[1]} columns (expected {EXPECTED_COLS})")
    missing_cols = [c for c in SCHEMA if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in SCHEMA]
    add("schema_columns", "*", len(missing_cols) + len(extra_cols),
        f"missing={missing_cols} extra={extra_cols}")

    for col, expected in SCHEMA.items():
        if col not in df.columns:
            continue
        s = df[col]
        actual, ok, note = type_check(s, expected)
        add("type", col, int(not ok), f"expected {expected}, got {actual}" + (f"; {note}" if note else ""))

        miss = missing_mask(s)
        n_nan, n_q = int(s.isna().sum()), int(miss.sum() - s.isna().sum())
        add("missing ('?' or NaN)", col, int(miss.sum()), f"NaN={n_nan}, '?'={n_q}")

        if col in RANGES:
            lo, hi = RANGES[col]
            v = pd.to_numeric(s, errors="coerce")
            bad = ((v < lo) | (v > hi)) & v.notna()
            add("range", col, int(bad.sum()),
                f"allowed {lo}-{hi}; observed {v.min():g}-{v.max():g}")

        if expected == "category" and col in DICTIONARY:
            vals = clean_str(s)
            allowed = set(DICTIONARY[col]) | {"?"}
            unknown = vals.notna() & ~vals.isin(allowed)
            add("allowed_categories", col, int(unknown.sum()),
                "unknown: " + ", ".join(sorted(map(str, vals[unknown].unique()))[:10]) if unknown.any()
                else f"{vals.nunique()} distinct values, all in data dictionary")

    feats = [c for c in df.columns if c != "income"]
    add("duplicate_rows (all columns)", "*", int(df.duplicated().sum()), "reported, not dropped")
    add("duplicate_feature_rows", "*", int(df.duplicated(subset=feats).sum()),
        "same 14 features (label may differ)")

    report = pd.DataFrame(rows)
    report.to_csv(RESULTS / "validation_report.csv", index=False)
    failed = report[report["violations"] > 0]
    summary = {
        "checks_run": int(len(report)),
        "checks_with_violations": int(len(failed)),
        "rows_with_any_missing": int(pd.concat([missing_mask(df[c]) for c in df.columns], axis=1)
                                     .any(axis=1).sum()),
        "missing_as_nan": int(sum(df[c].isna().sum() for c in df.columns)),
        "missing_as_question_mark": int(sum((missing_mask(df[c]) & df[c].notna()).sum() for c in df.columns)),
        "rows_dropped": 0,
    }
    (RESULTS / "validation_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"{summary['checks_run']} checks, {summary['checks_with_violations']} with violations "
          f"(no rows dropped):")
    print(failed.to_string(index=False) if len(failed) else "  none")


if __name__ == "__main__":
    main()
