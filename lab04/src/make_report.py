import json
 
import pandas as pd
import yaml
 
from common import LAB_DIR, RESULTS, clean_str, load_raw, schema_view
 
 
def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.astype(object).itertuples(index=False):
        out.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(out)
 
 
def pct(x):
    return f"{100 * x:.1f}%"
 
 
def group_stats(df):
    """Shares and >50K rates by sex, race and native country (for the data card)."""
    pos = clean_str(df["income"]) == ">50K"
    out = {"positive_rate": pos.mean()}
    sex = clean_str(df["sex"])
    out["male_share"] = (sex == "Male").mean()
    out["pos_male"] = pos[sex == "Male"].mean()
    out["pos_female"] = pos[sex == "Female"].mean()
    race = clean_str(df["race"])
    out["white_share"] = (race == "White").mean()
    out["race_table"] = (pd.DataFrame({"race": race, "pos": pos})
                         .groupby("race")["pos"].agg(["size", "mean"]).sort_values("size", ascending=False))
    nc = clean_str(df["native-country"])
    out["us_share"] = (nc == "United-States").mean()
    out["nc_missing"] = (nc.isna() | (nc == "?")).mean()
    return out
 
 
def main():
    ing = json.loads((RESULTS / "ingest_log.json").read_text())
    vsum = json.loads((RESULTS / "validation_summary.json").read_text())
    vrep = pd.read_csv(RESULTS / "validation_report.csv").fillna("")
    clean = json.loads((RESULTS / "cleaning_log.json").read_text())
    stats = pd.read_csv(RESULTS / "fitted_numeric_stats.csv")
    feat = pd.read_csv(RESULTS / "feature_eval.csv")
    audit = json.loads((RESULTS / "label_audit.json").read_text())
    cfg = yaml.safe_load((RESULTS / "config.yaml").read_text())
    py = (RESULTS / "versions.txt").read_text().splitlines()[0]
    g = group_stats(schema_view(load_raw()))
 
    base, der = feat.iloc[0], feat.iloc[1]
    d_acc = der["accuracy"] - base["accuracy"]
    d_f1 = der["f1 (>50K)"] - base["f1 (>50K)"]
    viol = vrep[vrep["violations"] > 0][["check", "column", "violations", "detail"]]
    miss = vrep[vrep["check"].str.startswith("missing") & (vrep["violations"] > 0)]
    miss_txt = ", ".join(f"{r.column} ({r.violations})" for r in miss.itertuples())
    stats_fmt = stats.copy()
    for c in stats_fmt.columns[1:]:
        stats_fmt[c] = stats_fmt[c].map(lambda v: f"{v:.4g}")
 
    n_nan, n_q = vsum["missing_as_nan"], vsum["missing_as_question_mark"]
    if n_nan and not n_q:
        null_concl = (f"Missing values were only in {miss_txt or 'no columns'}. The manual says missing "
                      f"values look like '?', but in the OpenML version all {n_nan} of them are already NaN. "
                      "If I had only searched for '?', I would have found zero missing values. So a data "
                      "check must follow the rule (what counts as missing), not one exact symbol.")
    else:
        null_concl = (f"Missing values were in {miss_txt or 'no columns'} ({n_nan} as NaN and {n_q} as '?'). "
                      "I counted both as missing, as the manual says. Otherwise the number of missing "
                      "values, and the imputation, would be wrong.")
    if abs(d_f1) < 0.005:
        feat_concl = (f"The two new features almost did not change the result (accuracy {d_acc:+.4f}, "
                      f"F1 {d_f1:+.4f}). capital_net is just capital-gain minus capital-loss, and the model "
                      "already sees both of them, so it learns nothing new. A new feature is only useful "
                      "if the test shows it helps.")
    elif d_f1 > 0:
        feat_concl = (f"The two new features made the model a little better (accuracy {d_acc:+.4f}, "
                      f"F1 {d_f1:+.4f}). A linear model cannot divide one column by another by itself, so "
                      "hours_per_age gives it new information. The features are calculated row by row, "
                      "so they do not leak test data.")
    else:
        feat_concl = (f"The two new features made the model a little worse (accuracy {d_acc:+.4f}, "
                      f"F1 {d_f1:+.4f}). So a new feature should not be kept just because it sounds useful; "
                      "it must be tested first.")
 
    report = f"""# Lab 4: Data pipeline and quality
 
## 1. Goal
 
The goal of this lab is to build a data pipeline for the UCI Adult dataset that:
 
- loads the raw data without changing it,
- checks the data for errors (wrong types, values out of range, missing values, duplicates),
- cleans the data **without leakage** (the test set never influences the cleaning),
- tests two new features,
- checks the quality of the labels,
- can be run again with **one command** and gives the same result.
 
### Problem definition
 
**Task.** Predict if a person earns **more than $50K per year** (yes or no). This is binary classification.
 
**Inputs.** 14 features about a person: age, workclass, fnlwgt (census weight), education,
education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss,
hours-per-week and native-country. 6 of them are numbers and 8 are categories (text).
 
**Output.** One label per person: `>50K` or `<=50K`.
 
**How success is measured.** Accuracy and F1 score on the test set (30% of the data).
I use F1 together with accuracy because the classes are not balanced: only {pct(g['positive_rate'])} of
people earn more than $50K. A model that always says "<=50K" would already get about
{pct(1 - g['positive_rate'])} accuracy, but it would be useless. F1 shows if the model can really find
the people who earn more than $50K.
 
**Rules (constraints).**
- No leakage: the data is split into train and test **first**. All values used for cleaning (median,
  most common value, percentiles, scaling) are learned from the train set only.
- Reproducible: fixed seed 42, fixed library versions, saved preprocessor, SHA256 hash of the processed
  data, and one command to run everything: `python lab04/src/run_all.py`.
 
## 2. Method
 
Environment: {py}. All library versions are in [results/versions.txt](results/versions.txt).
 
1. **Load the data.** I load Adult with `fetch_openml(name='adult', version=2, as_frame=True)`. The data is
   saved in `data/`. I take a hash of the data before and after loading to show that nothing was changed.
2. **Check the data.** I wrote down the expected type of every column, the allowed ranges
   (age 17-90, hours 1-99, capital-gain 0-99999, capital-loss 0-4356) and the rule that `'?'` means a
   missing value. I also check categories against the official UCI list and count duplicates.
   Problems are only reported. No rows are deleted at this step.
3. **Clean the data.**
   - First split into train (70%) and test (30%), stratified, seed 42.
   - Delete exact duplicate rows from the train set only. The test set stays as it is.
   - Numbers: fill missing values with the median, cut extreme values to the 1st and 99th percentile,
     then scale.
   - Categories: fill missing values with the most common value, then one-hot encode.
   - Everything is learned on train only and saved to `results/preprocessor.joblib`.
4. **New features.** `capital_net = capital-gain - capital-loss` and
   `hours_per_age = hours-per-week / (age - 16)`. I train `LogisticRegression(max_iter=1000, random_state=42)`
   with and without them and compare.
5. **Label check.** I look for people with exactly the same 14 features but different labels. I also take
   50 random rows and check them against the data dictionary (allowed values, education matches
   education-num, "Husband" is male, and so on).
6. **Save a fingerprint.** I compute a SHA256 hash of the processed train and test data and save it with
   all settings in `results/config.yaml`. The data card is in [data_card.md](data_card.md).
 
## 3. Results
 
### Loading
 
| Rows | Columns | Expected | Shape OK | Raw data unchanged |
|---|---|---|---|---|
| {ing['rows']} | {ing['columns']} | {ing['expected_rows']} x {ing['expected_columns']} | {ing['shape_ok']} | {ing['raw_unmodified']} |
 
### Data check (validation report)
 
I ran {vsum['checks_run']} checks. {vsum['checks_with_violations']} of them found problems.
{vsum['rows_with_any_missing']} rows have at least one missing value. No rows were deleted here.
The full report is in [results/validation_report.csv](results/validation_report.csv).
 
{md_table(viol) if len(viol) else "No problems found."}
 
### Cleaning (learned on train only)
 
| Train rows before | Duplicates deleted | Train rows after | Test rows (not changed) | Features after processing |
|---|---|---|---|---|
| {clean['train_rows_before_dedup']} | {clean['train_duplicates_dropped']} | {clean['train_rows']} | {clean['test_rows']} | {clean['processed_features']} |
 
Values learned from the train set for the number columns:
 
{md_table(stats_fmt)}
 
Most common values used to fill missing categories:
{", ".join(f"{k} = {v}" for k, v in clean['categorical_modes (train)'].items())}.
Number of values cut to the 1st/99th percentile in train: {clean['train_values_clipped']}.
 
### New features (LogisticRegression, test set)
 
{md_table(feat.drop(columns="derived"))}
 
### Label quality
 
| Rows with a duplicate feature set | Groups with different labels | Rows with different labels | Rows checked by hand | Rows with impossible values |
|---|---|---|---|---|
| {audit['rows_in_duplicate_feature_groups']} | {audit['conflicting_groups']} | {audit['rows_in_conflicting_groups']} | {audit['sample_size']} | {audit['sample_rows_with_impossible_values']} |
 
The 50 checked rows are in [results/label_audit_sample.csv](results/label_audit_sample.csv).
The rows with different labels are in [results/label_conflicts.csv](results/label_conflicts.csv).
 
### Reproducibility
 
| File | SHA256 |
|---|---|
| processed train | `{cfg['processed_sha256']['train']}` |
| processed test | `{cfg['processed_sha256']['test']}` |
 
All settings are in [results/config.yaml](results/config.yaml). To get all results again, run
`python lab04/src/run_all.py`.
 
## 4. Conclusions
 
1. {null_concl}
2. {feat_concl}
3. Even a well-known dataset has label errors: {audit['conflicting_groups']} groups of people
   ({audit['rows_in_conflicting_groups']} rows) have exactly the same features but different labels, so
   no model can be right on all of them. Splitting the data first and learning everything only from the
   train set keeps the test score honest. The saved hash and settings prove which data was used.
"""
    (LAB_DIR / "report.md").write_text(report, encoding="utf-8")
 
    race_tbl = g["race_table"].reset_index()
    race_tbl.columns = ["race", "rows", ">50K rate"]
    race_tbl[">50K rate"] = race_tbl[">50K rate"].map(pct)
    card = f"""# Data Card: UCI Adult / Census Income (Lab 4)
 
## Source
 
- **Dataset**: Adult (Census Income), extracted by Barry Becker from the 1994 US Census database;
  donated to the UCI Machine Learning Repository by Ronny Kohavi and Barry Becker (1996).
- **Access used here**: OpenML dataset 1590 via `fetch_openml(name='adult', version=2, as_frame=True)`,
  which combines the original UCI train (`adult.data`) and test (`adult.test`) files.
- **Citation**: Becker, B. & Kohavi, R. (1996). Adult. UCI Machine Learning Repository.
  https://doi.org/10.24432/C5XW20
 
## License
 
Creative Commons Attribution 4.0 International (CC BY 4.0), as listed on the UCI repository. Use requires
attribution to the creators.
 
## Size and structure
 
- {ing['rows']} rows x {ing['columns']} columns (14 features + income label).
- 6 numeric features (age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week) and
  8 categorical features (workclass, education, marital-status, occupation, relationship, race, sex,
  native-country).
- Label: `>50K` for {pct(g['positive_rate'])} of rows, `<=50K` for {pct(1 - g['positive_rate'])} (imbalanced).
- Missing values (`'?'`/NaN) occur in: {miss_txt or "none"}; see
  [results/validation_report.csv](results/validation_report.csv).
- Processed version: {clean['processed_features']} features after one-hot encoding; SHA256 of train
  `{cfg['processed_sha256']['train'][:16]}...` and test `{cfg['processed_sha256']['test'][:16]}...`
  (full values in [results/config.yaml](results/config.yaml)).
 
## Known biases
 
- **Sex**: {pct(g['male_share'])} of rows are male. The >50K rate is {pct(g['pos_male'])} for men and
  {pct(g['pos_female'])} for women, so a model can learn sex as a proxy for income.
- **Race**: {pct(g['white_share'])} of rows are White; other groups are small, so their error rates are
  estimated from few examples.
 
{md_table(race_tbl)}
 
- **Geography**: {pct(g['us_share'])} of rows list the United States as native country and
  {pct(g['nc_missing'])} are missing; most countries have only a handful of rows.
- **Time**: incomes are from 1994 and the $50K threshold is not inflation-adjusted, so the label does not
  describe today's income distribution.
- **Sampling**: the extraction kept only records with age > 16, adjusted gross income > 100,
  fnlwgt > 1 and hours-per-week > 0, so it is not a random sample of the population; fnlwgt is a census
  weight, not a personal attribute.
- **Label noise**: {audit['conflicting_groups']} groups of identical feature rows have conflicting labels.
 
## Privacy concerns
 
- The data is de-identified (no names or addresses), but it contains sensitive attributes (race, sex,
  marital status, relationship, national origin) and income.
- Combinations of quasi-identifiers such as age, sex, race, native-country and occupation can make a person
  unique in a small group, so re-identification risk exists when joined with other data.
- Models trained on it should not be used for real decisions about individuals (credit, hiring) without a
  fairness review (see Lab 11).
"""
    (LAB_DIR / "data_card.md").write_text(card, encoding="utf-8")
    print("Wrote report.md and data_card.md")
 
 
if __name__ == "__main__":
    main()