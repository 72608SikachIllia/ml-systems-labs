"""Step 5"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from common import RESULTS, SEED, build_preprocessor, load_raw, make_split, set_seeds


def main():
    set_seeds()
    X_tr, X_te, y_tr, y_te, _ = make_split(load_raw())
    rows = []
    for derived in (False, True):
        model = Pipeline([
            ("pre", build_preprocessor(derived=derived)),
            ("lr", LogisticRegression(max_iter=1000, random_state=SEED)),
        ]).fit(X_tr, y_tr)
        pred = model.predict(X_te)
        rows.append({
            "features": "14 original + capital_net + hours_per_age" if derived else "14 original",
            "derived": derived,
            "accuracy": round(accuracy_score(y_te, pred), 4),
            "f1 (>50K)": round(f1_score(y_te, pred), 4),
        })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "feature_eval.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
