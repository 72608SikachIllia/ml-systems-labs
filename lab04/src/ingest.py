"""Step 2"""
import json

from common import EXPECTED_COLS, EXPECTED_ROWS, RESULTS, frame_sha256, load_raw, target_column


def main():
    raw = load_raw()
    before = frame_sha256(raw)
    n_rows, n_cols = raw.shape
    tgt = target_column(raw)
    log = {
        "source": "fetch_openml(name='adult', version=2, as_frame=True)  # OpenML 1590",
        "rows": int(n_rows),
        "columns": int(n_cols),
        "expected_rows": EXPECTED_ROWS,
        "expected_columns": EXPECTED_COLS,
        "shape_ok": bool(n_rows == EXPECTED_ROWS and n_cols == EXPECTED_COLS),
        "column_names": list(map(str, raw.columns)),
        "dtypes": {str(c): str(t) for c, t in raw.dtypes.items()},
        "target_column": tgt,
        "label_counts": {str(k): int(v) for k, v in raw[tgt].value_counts(dropna=False).items()},
        "raw_sha256": before,
    }
    log["raw_unmodified"] = frame_sha256(raw) == before
    (RESULTS / "ingest_log.json").write_text(json.dumps(log, indent=2))
    print(f"Ingested {n_rows} rows x {n_cols} columns "
          f"(expected {EXPECTED_ROWS} x {EXPECTED_COLS}) -> shape_ok={log['shape_ok']}")
    print(f"Label column '{tgt}': {log['label_counts']}")


if __name__ == "__main__":
    main()
