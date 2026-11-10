# Lab 4: Data pipeline and quality
 
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
I use F1 together with accuracy because the classes are not balanced: only 23.9% of
people earn more than $50K. A model that always says "<=50K" would already get about
76.1% accuracy, but it would be useless. F1 shows if the model can really find
the people who earn more than $50K.
 
**Rules (constraints).**
- No leakage: the data is split into train and test **first**. All values used for cleaning (median,
  most common value, percentiles, scaling) are learned from the train set only.
- Reproducible: fixed seed 42, fixed library versions, saved preprocessor, SHA256 hash of the processed
  data, and one command to run everything: `python lab04/src/run_all.py`.
 
## 2. Method
 
Environment: Python 3.11.8 (CPython). All library versions are in [results/versions.txt](results/versions.txt).
 
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
| 48842 | 15 | 48842 x 15 | True | True |
 
### Data check (validation report)
 
I ran 48 checks. 5 of them found problems.
3620 rows have at least one missing value. No rows were deleted here.
The full report is in [results/validation_report.csv](results/validation_report.csv).
 
| check | column | violations | detail |
|---|---|---|---|
| missing ('?' or NaN) | workclass | 2799 | NaN=2799, '?'=0 |
| missing ('?' or NaN) | occupation | 2809 | NaN=2809, '?'=0 |
| missing ('?' or NaN) | native-country | 857 | NaN=857, '?'=0 |
| duplicate_rows (all columns) | * | 52 | reported, not dropped |
| duplicate_feature_rows | * | 57 | same 14 features (label may differ) |
 
### Cleaning (learned on train only)
 
| Train rows before | Duplicates deleted | Train rows after | Test rows (not changed) | Features after processing |
|---|---|---|---|---|
| 34189 | 24 | 34165 | 14653 | 107 |
 
Values learned from the train set for the number columns:
 
| feature | median (train) | clip_low p1 (train) | clip_high p99 (train) |
|---|---|---|---|
| age | 37 | 17 | 74 |
| fnlwgt | 1.783e+05 | 2.704e+04 | 5.096e+05 |
| education-num | 10 | 3 | 16 |
| capital-gain | 0 | 0 | 1.502e+04 |
| capital-loss | 0 | 0 | 1980 |
| hours-per-week | 40 | 8 | 80 |
| capital_net | 0 | -1980 | 1.502e+04 |
| hours_per_age | 2 | 0.2083 | 20 |
 
Most common values used to fill missing categories:
workclass = Private, education = HS-grad, marital-status = Married-civ-spouse, occupation = Prof-specialty, relationship = Husband, race = White, sex = Male, native-country = United-States.
Number of values cut to the 1st/99th percentile in train: 2357.
 
### New features (LogisticRegression, test set)
 
| features | accuracy | f1 (>50K) |
|---|---|---|
| 14 original | 0.854 | 0.6601 |
| 14 original + capital_net + hours_per_age | 0.8578 | 0.6696 |
 
### Label quality
 
| Rows with a duplicate feature set | Groups with different labels | Rows with different labels | Rows checked by hand | Rows with impossible values |
|---|---|---|---|---|
| 111 | 5 | 10 | 50 | 0 |
 
The 50 checked rows are in [results/label_audit_sample.csv](results/label_audit_sample.csv).
The rows with different labels are in [results/label_conflicts.csv](results/label_conflicts.csv).
 
### Reproducibility
 
| File | SHA256 |
|---|---|
| processed train | `d13dcaa68ab77881d6cb19fdb52f1fa28e71788413e87de5e225d457fffaafd3` |
| processed test | `3a50748c14ff6a7722e313d67867e9bcbd89e0944fe4a5ae2301ac2c0a72ecd3` |
 
All settings are in [results/config.yaml](results/config.yaml). To get all results again, run
`python lab04/src/run_all.py`.
 
## 4. Conclusions
 
1. Missing values were only in workclass (2799), occupation (2809), native-country (857). The manual says missing values look like '?', but in the OpenML version all 6465 of them are already NaN. If I had only searched for '?', I would have found zero missing values. So a data check must follow the rule (what counts as missing), not one exact symbol.
2. The two new features made the model a little better (accuracy +0.0038, F1 +0.0095). A linear model cannot divide one column by another by itself, so hours_per_age gives it new information. The features are calculated row by row, so they do not leak test data.
3. Even a well-known dataset has label errors: 5 groups of people
   (10 rows) have exactly the same features but different labels, so
   no model can be right on all of them. Splitting the data first and learning everything only from the
   train set keeps the test score honest. The saved hash and settings prove which data was used.
