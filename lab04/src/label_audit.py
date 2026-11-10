"""Step 6
"""
import json

import numpy as np
import pandas as pd

from common import (CATEGORICAL, DICTIONARY, EDUCATION_NUM, FEATURES, RANGES, RESULTS, SEED,
                    clean_str, load_raw, schema_view)


def normalised(raw):
    df = schema_view(raw)
    out = df[FEATURES + ["income"]].copy()
    for c in CATEGORICAL + ["income"]:
        out[c] = clean_str(out[c]).fillna("?")
    for c in FEATURES:
        if c not in CATEGORICAL:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def row_problems(r) -> list:
    p = []
    for col, (lo, hi) in RANGES.items():
        if pd.isna(r[col]) or not lo <= r[col] <= hi:
            p.append(f"{col}={r[col]} outside {lo}-{hi}")
    if pd.isna(r["fnlwgt"]) or r["fnlwgt"] <= 0:
        p.append(f"fnlwgt={r['fnlwgt']} not positive")
    for col in CATEGORICAL + ["income"]:
        if r[col] not in set(DICTIONARY[col]) | {"?"}:
            p.append(f"{col}='{r[col]}' not in dictionary")
    exp = EDUCATION_NUM.get(r["education"])
    if exp is not None and r["education-num"] != exp:
        p.append(f"education-num={r['education-num']} but {r['education']} = {exp}")
    if r["relationship"] == "Husband" and r["sex"] != "Male":
        p.append("Husband but not Male")
    if r["relationship"] == "Wife" and r["sex"] != "Female":
        p.append("Wife but not Female")
    if r["relationship"] in ("Husband", "Wife") and not str(r["marital-status"]).startswith("Married"):
        p.append(f"{r['relationship']} but marital-status={r['marital-status']}")
    if r["capital-gain"] > 0 and r["capital-loss"] > 0:
        p.append("both capital-gain and capital-loss > 0")
    return p


def main():
    df = normalised(load_raw())

    # (a) duplicate-conflict audit on the full dataset
    groups = df.groupby(FEATURES, dropna=False)["income"]
    n_labels = groups.transform("nunique")
    dup_feat = df.duplicated(subset=FEATURES, keep=False)
    conflict_rows = df[n_labels > 1]
    n_conflict_groups = int((groups.nunique() > 1).sum())
    conflict_rows.sort_values(FEATURES).to_csv(RESULTS / "label_conflicts.csv", index=False)

    # (b) 50 random rows against the data dictionary
    sample = df.sample(n=50, random_state=SEED).copy()
    sample["problems"] = sample.apply(lambda r: "; ".join(row_problems(r)), axis=1)
    sample["ok"] = sample["problems"] == ""
    sample.to_csv(RESULTS / "label_audit_sample.csv", index_label="row_id")

    summary = {
        "rows_in_duplicate_feature_groups": int(dup_feat.sum()),
        "conflicting_groups": n_conflict_groups,
        "rows_in_conflicting_groups": int(len(conflict_rows)),
        "conflict_rate_of_all_rows": round(len(conflict_rows) / len(df), 6),
        "sample_size": 50,
        "sample_rows_with_impossible_values": int((~sample["ok"]).sum()),
        "sample_rows_with_missing_('?')": int((sample[CATEGORICAL] == "?").any(axis=1).sum()),
        "sample_label_counts": {str(k): int(v) for k, v in sample["income"].value_counts().items()},
    }
    (RESULTS / "label_audit.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    bad = sample.loc[~sample["ok"], "problems"]
    if len(bad):
        print("Sample rows with problems:\n" + bad.to_string())


if __name__ == "__main__":
    main()
