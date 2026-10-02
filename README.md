# Bayesian Network with SMOTE for COVID-19 Diagnosis on Imbalanced Data

Code and results for the paper:

> Asif Ali, Sanam Narejo, Shahnawaz Talpur, Fawad Ali Mangi. *Uncertainty-Aware Bayesian Network Diagnosis for Imbalanced Pandemic Data: A COVID-19 Case Study.* Journal of Computational Science and Applications (submitted).

The notebook trains a Bayesian Network on symptom and demographic data, balances the classes with SMOTE, compares the model with Logistic Regression, KNN and Random Forest, and validates it on data from a later period of the pandemic (temporal validation).

## Main results

| Metric | Internal test set (Mar–Apr 2020) | Temporal validation set (May–Nov 2020) |
|---|---|---|
| Accuracy | 0.9127 | 0.9340 |
| Precision | 0.3559 | 0.6108 |
| Recall (Sensitivity) | 0.7802 | 0.6266 |
| Specificity | 0.9202 | 0.9627 |
| F1-score | 0.4888 | 0.6186 |
| AUC | 0.9011 | 0.8177 |

* SMOTE increased the sensitivity of the Bayesian Network from 54.10% to 78.02% on the internal test set.
* Random Forest reached the same AUC (0.90), but needed about 6 s for training and prediction, against about 0.2 s for the Bayesian Network.

All tables are in [`results/`](results) and all figures in [`figures/`](figures).

## Repository structure

```
BN_SMOTE_temporal_validation.ipynb   full experiment (already executed, outputs included)
requirements.txt                     Python packages
data/                                the two dataset versions are downloaded here on the first run
results/                             result tables (CSV)
figures/                             result figures (PNG, 300 dpi)
```

## How to run

```bash
git clone https://github.com/narejohumair-zt/COVID19-BN-SMOTE.git
cd COVID19-BN-SMOTE
pip install -r requirements.txt
jupyter notebook BN_SMOTE_temporal_validation.ipynb
```

Run all cells. On the first run, the notebook downloads the two dataset files into `data/`. The full run takes about 2 minutes on a normal laptop.

## Data

Both datasets come from the public CovidPred repository of Zoabi et al.: https://github.com/nshomron/covidpred

| File | Downloaded | Test dates | Used as |
|---|---|---|---|
| `corona_tested_individuals_ver_006.english.csv.zip` | 4 May 2020 | 11 Mar – 30 Apr 2020 | development data (80% training, 20% internal test) |
| `corona_tested_individuals_ver_0083.english.csv.zip` | 15 Nov 2020 | 11 Mar – 12 Nov 2020 | temporal validation (only records tested after 30 Apr 2020) |

The second version also contains the March and April records, so only its records after 30 April 2020 are used for validation. No record of the validation period is used in training. The data are not stored in this repository; please cite the original authors when using them:

> Zoabi, Y., Deri-Rozov, S. and Shomron, N. (2021). Machine learning-based prediction of COVID-19 diagnosis based on symptoms. *npj Digital Medicine*, 4, 3.

## Method summary

1. Remove records with result `other` and records with missing symptoms.
2. Encode the features (gender: female 0 / male 1; test indication: contact with confirmed 1, abroad 2, other 3).
3. Stratified 80/20 split of the development data.
4. Fill missing `age_60_and_above` and `gender` with linear regression fitted on the training set only.
5. Balance the training set with SMOTE-N (the nominal version of SMOTE). The test sets keep their real class ratio.
6. Bayesian Network: all eight features point to `corona_result`; parameters by Maximum Likelihood Estimation; inference by Variable Elimination (pgmpy).
7. Baselines: Logistic Regression, KNN (k = 5), Random Forest (100 trees), and the same Bayesian Network without SMOTE.
8. Evaluation on the internal test set and on the temporal validation set (accuracy, precision, sensitivity, specificity, F1-score for the positive class, and AUC from predicted probabilities).

## Notes

* All metrics are fixed by the random seed (42). Running times depend on the computer; each reported time is the mean of 5 runs.
* For every model, predictions are computed once for each distinct combination of feature values and mapped back to the records. This gives the same predictions as record-by-record prediction and keeps inference on 2.4 million records fast.
* The included outputs were produced with Python 3.13, pandas 3.0, scikit-learn 1.9, imbalanced-learn 0.15, pgmpy 1.0.0 and matplotlib 3.11.

## License

The code is released under the MIT License (see [`LICENSE`](LICENSE)). The datasets belong to their original authors.
