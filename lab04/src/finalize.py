"""Step 7"""
import json
import platform

import yaml

from common import (CATEGORICAL, CLIP_PERCENTILES, DERIVED, NUMERIC, PROCESSED_DIR, REPO_ROOT,
                    RESULTS, SEED, TEST_SIZE, sha256_file)


def main():
    files = {name: PROCESSED_DIR / f"{name}.csv" for name in ("train", "test")}
    hashes = {name: sha256_file(p) for name, p in files.items()}
    ingest = json.loads((RESULTS / "ingest_log.json").read_text())
    clean = json.loads((RESULTS / "cleaning_log.json").read_text())

    config = {
        "dataset": {
            "name": "UCI Adult / Census Income",
            "loader": "fetch_openml(name='adult', version=2, as_frame=True)",
            "openml_id": 1590,
            "raw_rows": ingest["rows"],
            "raw_columns": ingest["columns"],
            "raw_sha256": ingest["raw_sha256"],
        },
        "seed": SEED,
        "split": {"test_size": TEST_SIZE, "stratify": "income", "random_state": SEED,
                  "done_before_cleaning": True},
        "cleaning": {
            "null_policy": "'?' and NaN are missing",
            "duplicates": "exact duplicates dropped from train only",
            "train_duplicates_dropped": clean["train_duplicates_dropped"],
            "imputation": {"numeric": "median (fit on train)",
                           "categorical": "most_frequent (fit on train)"},
            "outliers": f"clip numeric to p{CLIP_PERCENTILES[0]}/p{CLIP_PERCENTILES[1]} of train",
            "scaling": "StandardScaler (fit on train)",
            "encoding": "OneHotEncoder(handle_unknown='ignore')",
        },
        "features": {"numeric": NUMERIC, "categorical": CATEGORICAL, "derived": DERIVED,
                     "processed_feature_count": clean["processed_features"]},
        "artifacts": {
            "preprocessor": "lab04/results/preprocessor.joblib",
            "processed_train": str(files["train"].relative_to(REPO_ROOT)).replace("\\", "/"),
            "processed_test": str(files["test"].relative_to(REPO_ROOT)).replace("\\", "/"),
        },
        "processed_sha256": hashes,
        "python": platform.python_version(),
    }
    (RESULTS / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (RESULTS / "processed_sha256.txt").write_text(
        "".join(f"{h}  {config['artifacts']['processed_' + n]}\n" for n, h in hashes.items()))
    for n, h in hashes.items():
        print(f"SHA256 processed {n}: {h}")
    print("Saved results/config.yaml")


if __name__ == "__main__":
    main()
