"""Shared helpers for Lab 4: seeds, paths, schema, data dictionary, split and preprocessor."""
import hashlib
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

SEED = 42
TEST_SIZE = 0.30
CLIP_PERCENTILES = (1, 99)

LAB_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_DIR.parent
RESULTS = LAB_DIR / "results"
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed" / "lab04"
for d in (RESULTS, DATA_DIR, PROCESSED_DIR):
    d.mkdir(parents=True, exist_ok=True)

EXPECTED_ROWS = 48842
EXPECTED_COLS = 15
POSITIVE_LABEL = ">50K"

# schema 
SCHEMA = {
    "age": "int",
    "workclass": "category",
    "fnlwgt": "int",
    "education": "category",
    "education-num": "int",
    "marital-status": "category",
    "occupation": "category",
    "relationship": "category",
    "race": "category",
    "sex": "category",
    "capital-gain": "int",
    "capital-loss": "int",
    "hours-per-week": "int",
    "native-country": "category",
    "income": "category",
}
RANGES = {
    "age": (17, 90),
    "hours-per-week": (1, 99),
    "capital-gain": (0, 99999),
    "capital-loss": (0, 4356),
}
FEATURES = [c for c in SCHEMA if c != "income"]
NUMERIC = [c for c in FEATURES if SCHEMA[c] == "int"]
CATEGORICAL = [c for c in FEATURES if SCHEMA[c] == "category"]
DERIVED = ["capital_net", "hours_per_age"]

# UCI data dictionary 
DICTIONARY = {
    "workclass": ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov",
                  "State-gov", "Without-pay", "Never-worked"],
    "education": ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm",
                  "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate",
                  "5th-6th", "Preschool"],
    "marital-status": ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed",
                       "Married-spouse-absent", "Married-AF-spouse"],
    "occupation": ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
                   "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
                   "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv",
                   "Armed-Forces"],
    "relationship": ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"],
    "race": ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
    "sex": ["Female", "Male"],
    "native-country": ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany",
                       "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China",
                       "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica",
                       "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic",
                       "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala",
                       "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador",
                       "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"],
    "income": [">50K", "<=50K"],
}
EDUCATION_NUM = {
    "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4, "9th": 5, "10th": 6, "11th": 7,
    "12th": 8, "HS-grad": 9, "Some-college": 10, "Assoc-voc": 11, "Assoc-acdm": 12,
    "Bachelors": 13, "Masters": 14, "Prof-school": 15, "Doctorate": 16,
}


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ingestion
def load_raw() -> pd.DataFrame:
    """Raw Adult frame exactly as OpenML returns it (cached in data/). Never modified in place."""
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(name="adult", version=2, as_frame=True, data_home=str(DATA_DIR))
    return bunch.frame.copy()


def target_column(df: pd.DataFrame) -> str:
    """OpenML names the label 'class'; the manual calls it 'income'."""
    for name in ("income", "class"):
        if name in df.columns:
            return name
    raise KeyError("No income/class column in the dataset")


def schema_view(df: pd.DataFrame) -> pd.DataFrame:
    """Same data with the label column named 'income' (a renamed copy, raw is untouched)."""
    return df.rename(columns={target_column(df): "income"})


def clean_str(s: pd.Series) -> pd.Series:
    """Category values as plain strings ('?' kept, NaN kept as NaN); strips spaces and trailing '.'."""
    out = s.astype(object)
    mask = out.notna()
    out[mask] = out[mask].astype(str).str.strip().str.rstrip(".")
    return out


def missing_mask(s: pd.Series) -> pd.Series:
    """Null policy: '?' means missing. OpenML already stores most of them as NaN."""
    return s.isna() | (s.astype(object).astype(str).str.strip() == "?")


def to_xy(raw: pd.DataFrame):
    """Features with '?' -> NaN (stateless, row-wise) and binary label 1 = '>50K'."""
    df = schema_view(raw)
    X = df[FEATURES].copy()
    for c in CATEGORICAL:
        X[c] = clean_str(X[c]).replace("?", np.nan)
    for c in NUMERIC:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    y = (clean_str(df["income"]) == POSITIVE_LABEL).astype(int)
    return X, y


def make_split(raw: pd.DataFrame):
    """70/30 stratified split BEFORE any cleaning; then exact duplicates dropped from train only."""
    X, y = to_xy(raw)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    train = X_tr.assign(_y=y_tr)
    dup = train.duplicated(keep="first")
    train = train[~dup]
    info = {"train_rows_before_dedup": int(len(X_tr)), "train_duplicates_dropped": int(dup.sum()),
            "train_rows": int(len(train)), "test_rows": int(len(X_te))}
    return train.drop(columns="_y"), X_te, train["_y"], y_te, info


#  preprocessing
def add_derived(X: pd.DataFrame) -> pd.DataFrame:
    """Row-wise derived features (no fitted statistics, so no leakage)."""
    X = X.copy()
    X["capital_net"] = X["capital-gain"] - X["capital-loss"]
    X["hours_per_age"] = X["hours-per-week"] / (X["age"] - 16)
    return X


class PercentileClipper(BaseEstimator, TransformerMixin):
    """Clip each numeric column to percentiles learned on the training data only."""

    def __init__(self, lower=CLIP_PERCENTILES[0], upper=CLIP_PERCENTILES[1]):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lower_ = np.nanpercentile(X, self.lower, axis=0)
        self.upper_ = np.nanpercentile(X, self.upper, axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        return np.asarray(input_features, dtype=object)


def build_preprocessor(derived: bool = True) -> Pipeline:
    numeric = NUMERIC + (DERIVED if derived else [])
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clip", PercentileClipper()),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    columns = ColumnTransformer([("num", num_pipe, numeric), ("cat", cat_pipe, CATEGORICAL)])
    steps = [("derive", FunctionTransformer(add_derived))] if derived else []
    return Pipeline(steps + [("columns", columns)])


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frame_sha256(df: pd.DataFrame) -> str:
    """Content hash of a DataFrame (used to prove ingestion did not modify the raw data)."""
    row_hashes = pd.util.hash_pandas_object(df, index=True).values
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()
