# Uncertainty-Aware Bayesian Network Diagnosis for Imbalanced Pandemic Data: A COVID-19 Case Study

Code for the paper by Asif Ali, Sanam Narejo, Shahnawaz Talpur and Fawad Ali Mangi
(Department of Computer Systems Engineering, Mehran University of Engineering & Technology, Jamshoro, Pakistan).

The project builds a Bayesian Network that predicts a COVID-19 test result from symptoms and
demographic information. Because positive cases are far fewer than negative ones, the classes are
balanced with SMOTE before the network parameters are learned. The model is compared with
Logistic Regression, KNN and Random Forest, and is then validated on an independent dataset.

## Method

1. **Preprocessing** – records with missing symptom values and records with the result "other" are
   removed, categorical values are encoded as integers, `test_date` is converted to the number of
   days since the first test, and missing `age_60_and_above` and `gender` values are filled with a
   regression model trained on the records where they are known.
2. **Class balancing** – SMOTE (`random_state=42`) oversamples the positive class.
3. **Bayesian Network** – every feature is a parent of `corona_result`; the conditional probability
   tables are estimated by Maximum Likelihood Estimation and predictions use Variable Elimination.
4. **Comparison** – Logistic Regression, KNN (k = 5) and Random Forest (100 trees) are trained on the
   same data. Accuracy, precision, recall, specificity, F1-score and total time are reported.
5. **Probabilistic inference** – posterior probability of a positive result for several symptom
   combinations, e.g. P(corona_result | cough = 1, fever = 1, sore_throat = 1).
6. **Independent validation** – a network built on the five symptoms shared by both datasets is
   trained on the balanced training data and tested on the independent dataset.

## Repository structure

```
├── main.py                    # runs the complete experiment
├── download_data.py           # downloads the training/test files
├── requirements.txt
├── data/
│   └── README.md              # data sources and the expected format of cmc.csv
├── results/                   # tables and figures are written here
└── src/
    ├── config.py              # paths, feature names, network structures, evidence sets
    ├── data_preprocessing.py  # cleaning, encoding, date conversion, imputation
    ├── balancing.py           # SMOTE
    ├── bayesian_network.py    # network construction, MLE fitting, prediction, queries
    ├── baselines.py           # Logistic Regression, KNN, Random Forest
    ├── evaluation.py          # metrics, ROC/AUC, timing
    ├── external_validation.py # loading of the independent dataset
    └── plots.py               # figures
```

## Installation

Python 3.9 or newer is required.

```bash
git clone <repository-url>
cd bn-smote-covid19
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The code works with both the older (0.1.x) and the newer (1.x) pgmpy releases.

## Data

**Training and test data.** The two files come from the public CovidPred repository
([nshomron/covidpred](https://github.com/nshomron/covidpred), Zoabi et al., 2021):

```bash
python download_data.py
```

This saves `corona_tested_individuals_ver_0083.english.csv.zip` (training) and
`corona_tested_individuals_ver_006.english.csv.zip` (test) in `data/`. The files can also be
downloaded by hand from the repository's `data/` folder; zipped or unzipped files both work.

**Independent validation data.** Place the independent dataset (Chandka Medical College Hospital,
Larkana) at `data/cmc.csv`. The expected columns are described in [data/README.md](data/README.md).
If the file is not present, the validation step is skipped.

## Usage

```bash
python main.py                      # complete experiment
python main.py --skip-baselines     # Bayesian Network only (faster)
python main.py --sample-frac 0.05   # quick check on 5% of the records
python main.py --skip-external      # skip the independent validation
python main.py --keep-test-distribution   # evaluate on the test file without SMOTE
```

On the full data the Bayesian Network step takes a few minutes; KNN and Random Forest take
considerably longer. About 8 GB of RAM is recommended.

## Outputs

| File in `results/` | Content |
|---|---|
| `class_distribution.csv` | class counts in the training file (Table 1) |
| `model_performance_and_times.csv` | metrics and total time of each model (Table 4) |
| `performance_metrics_chart.png` | metric comparison (Figure 6) |
| `total_time_chart.png` | training + prediction time (Figure 7) |
| `roc_curve.png` | ROC curve of the Bayesian Network (Figure 9) |
| `inference_example.png`, `inference_evidence_sets.png`, `inference_results.csv` | probabilistic inference examples (Figures 4 and 5) |
| `training_vs_independent.csv`, `training_vs_independent.png` | independent validation (Table 5, Figure 8) |
| `confusion_matrix_independent.png` | confusion matrix on the independent dataset |

Notes on the metrics: precision, recall and F1-score on the test data are weighted averages over
both classes; on the independent dataset they refer to the positive class. Specificity is
TN / (TN + FP). The ROC curve is computed from the predicted class labels. Exact values can vary
slightly between library versions.

## Citation

If you use this code, please cite:

```bibtex
@article{ali2026bayesian,
  title   = {Uncertainty-Aware Bayesian Network Diagnosis for Imbalanced Pandemic Data: A COVID-19 Case Study},
  author  = {Ali, Asif and Narejo, Sanam and Talpur, Shahnawaz and Mangi, Fawad Ali},
  journal = {Journal of Computational Science and Applications},
  year    = {2026},
  note    = {Manuscript submitted}
}
```

Please also cite the source of the training data:

> Y. Zoabi, S. Deri-Rozov, and N. Shomron, "Machine learning-based prediction of COVID-19
> diagnosis based on symptoms," *npj Digital Medicine*, vol. 4, no. 1, p. 3, 2021.
> doi: 10.1038/s41746-020-00372-6

## License

Released under the MIT License (see [LICENSE](LICENSE)). The datasets are subject to the terms of
their original sources.
