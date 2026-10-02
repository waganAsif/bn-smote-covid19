"""Paths, feature names and fixed settings used across the project."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# Training data (CovidPred public repository). Either the .csv or the .csv.zip
# file can be placed in data/.
TRAIN_FILE = "corona_tested_individuals_ver_0083.english.csv"
TEST_FILE = "corona_tested_individuals_ver_006.english.csv"
COVIDPRED_RAW_URL = "https://github.com/nshomron/covidpred/raw/master/data/{name}.zip"

# Independent validation dataset (CMC Hospital). Added by the authors.
CMC_FILE = DATA_DIR / "cmc.csv"

RANDOM_STATE = 42
TARGET = "corona_result"
SYMPTOMS = ["cough", "fever", "sore_throat", "shortness_of_breath", "head_ache"]

# Categorical value encoding (Table 2 of the paper)
CATEGORY_MAPS = {
    "corona_result": {"negative": 0, "positive": 1},
    "age_60_and_above": {"Yes": 1, "No": 0},
    "gender": {"male": 1, "female": 0},
    "test_indication": {"Contact with confirmed": 1, "Abroad": 2, "Other": 3},
}

# Network used for training/testing and the model comparison (Table 4)
EVAL_PARENTS = SYMPTOMS + ["age_60_and_above", "test_indication", "gender", "test_date_numeric"]

# Network used for the probabilistic inference examples (Figure 3 structure)
INFERENCE_PARENTS = SYMPTOMS + ["age_60_and_above", "test_indication", "gender"]

# Network used for independent validation: only the symptoms that are
# available in both datasets
EXTERNAL_PARENTS = list(SYMPTOMS)

# Column names in the CMC file and their names in the training data
CMC_COLUMN_MAP = {
    "Breathing Problem": "shortness_of_breath",
    "Fever": "fever",
    "Dry Cough": "cough",
    "Sore throat": "sore_throat",
    "Headache": "head_ache",
    "COVID-19": "corona_result",
}

# Evidence sets used for the inference examples (Figures 4 and 5)
EVIDENCE_EXAMPLE = {"cough": 1, "fever": 1, "sore_throat": 1}
EVIDENCE_SETS = [
    {"cough": 1, "fever": 1, "shortness_of_breath": 1, "sore_throat": 1, "head_ache": 1},
    {"cough": 0, "fever": 0, "shortness_of_breath": 0, "sore_throat": 0, "head_ache": 0},
    {"cough": 1, "fever": 1, "head_ache": 1},
    {"cough": 0, "fever": 0, "head_ache": 0},
]
