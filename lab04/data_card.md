# Data Card: UCI Adult / Census Income (Lab 4)
 
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
 
- 48842 rows x 15 columns (14 features + income label).
- 6 numeric features (age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week) and
  8 categorical features (workclass, education, marital-status, occupation, relationship, race, sex,
  native-country).
- Label: `>50K` for 23.9% of rows, `<=50K` for 76.1% (imbalanced).
- Missing values (`'?'`/NaN) occur in: workclass (2799), occupation (2809), native-country (857); see
  [results/validation_report.csv](results/validation_report.csv).
- Processed version: 107 features after one-hot encoding; SHA256 of train
  `d13dcaa68ab77881...` and test `3a50748c14ff6a77...`
  (full values in [results/config.yaml](results/config.yaml)).
 
## Known biases
 
- **Sex**: 66.8% of rows are male. The >50K rate is 30.4% for men and
  10.9% for women, so a model can learn sex as a proxy for income.
- **Race**: 85.5% of rows are White; other groups are small, so their error rates are
  estimated from few examples.
 
| race | rows | >50K rate |
|---|---|---|
| White | 41762 | 25.4% |
| Black | 4685 | 12.1% |
| Asian-Pac-Islander | 1519 | 26.9% |
| Amer-Indian-Eskimo | 470 | 11.7% |
| Other | 406 | 12.3% |
 
- **Geography**: 89.7% of rows list the United States as native country and
  1.8% are missing; most countries have only a handful of rows.
- **Time**: incomes are from 1994 and the $50K threshold is not inflation-adjusted, so the label does not
  describe today's income distribution.
- **Sampling**: the extraction kept only records with age > 16, adjusted gross income > 100,
  fnlwgt > 1 and hours-per-week > 0, so it is not a random sample of the population; fnlwgt is a census
  weight, not a personal attribute.
- **Label noise**: 5 groups of identical feature rows have conflicting labels.
 
## Privacy concerns
 
- The data is de-identified (no names or addresses), but it contains sensitive attributes (race, sex,
  marital status, relationship, national origin) and income.
- Combinations of quasi-identifiers such as age, sex, race, native-country and occupation can make a person
  unique in a small group, so re-identification risk exists when joined with other data.
- Models trained on it should not be used for real decisions about individuals (credit, hiring) without a
  fairness review (see Lab 11).
