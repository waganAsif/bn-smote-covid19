"""Loading and preprocessing of the training and test data.

Steps (Section 3 of the paper):
1. drop records with missing symptom values,
2. remove records whose corona_result is "other",
3. encode categorical values as integers,
4. convert test_date to days since the earliest test date,
5. fill missing age_60_and_above and gender values with a regression model
   trained on the records where the value is known.
"""

import zipfile
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from . import config


def find_data_file(name: str, data_dir: Path = config.DATA_DIR) -> Path:
    """Return the path of `name` in data/, accepting a .csv or a .csv.zip file."""
    for candidate in (data_dir / name, data_dir / f"{name}.zip"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"'{name}' was not found in {data_dir}. Run `python download_data.py` "
        "or see data/README.md for download instructions."
    )


def read_csv_any(path: Path, **kwargs) -> pd.DataFrame:
    """Read a .csv file, or the .csv inside a .zip archive (ignoring __MACOSX entries)."""
    if path.suffix != ".zip":
        return pd.read_csv(path, **kwargs)
    with zipfile.ZipFile(path) as archive:
        members = [m for m in archive.namelist()
                   if m.endswith(".csv") and not m.startswith("__MACOSX")]
        if not members:
            raise ValueError(f"No CSV file found inside {path}")
        with archive.open(members[0]) as handle:
            return pd.read_csv(handle, **kwargs)


def load_raw_data():
    """Load the raw training and test files."""
    train = read_csv_any(find_data_file(config.TRAIN_FILE), low_memory=False)
    test = read_csv_any(find_data_file(config.TEST_FILE), na_values="None", low_memory=False)
    return train, test


def clean_and_encode(df: pd.DataFrame) -> pd.DataFrame:
    """Drop incomplete symptom records, remove 'other' results and encode categories."""
    df = df.dropna(subset=config.SYMPTOMS).copy()
    df[config.TARGET] = df[config.TARGET].astype(str).str.strip().str.lower()
    df = df[df[config.TARGET] != "other"].copy()
    for column, mapping in config.CATEGORY_MAPS.items():
        df[column] = df[column].map(mapping).astype("float64")
    for column in config.SYMPTOMS:
        df[column] = df[column].astype("float64")
    return df


def add_test_date_numeric(train: pd.DataFrame, test: pd.DataFrame):
    """Add test_date_numeric: days since the earliest test date in both files."""
    train = train.copy()
    test = test.copy()
    train["test_date"] = pd.to_datetime(train["test_date"])
    test["test_date"] = pd.to_datetime(test["test_date"])
    earliest = min(train["test_date"].min(), test["test_date"].min())
    train["test_date_numeric"] = (train["test_date"] - earliest).dt.days
    test["test_date_numeric"] = (test["test_date"] - earliest).dt.days
    return train, test


def impute_column(df: pd.DataFrame, column: str, predictors: list, verbose: bool = True) -> pd.DataFrame:
    """Fill missing values of `column` with a logistic regression model."""
    missing = df[column].isna()
    if not missing.any():
        return df
    known = df.loc[~missing]
    X_train, X_val, y_train, y_val = train_test_split(
        known[predictors], known[column], test_size=0.2, random_state=config.RANDOM_STATE
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    if verbose:
        acc = accuracy_score(y_val, model.predict(X_val))
        print(f"  imputation of '{column}': {missing.sum():,} missing values, "
              f"validation accuracy {acc:.4f}")
    df = df.copy()
    df.loc[missing, column] = model.predict(df.loc[missing, predictors])
    return df


def impute_demographics(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Impute age_60_and_above first, then gender (which also uses the imputed age)."""
    predictors = ["test_date_numeric"] + config.SYMPTOMS + [config.TARGET, "test_indication"]
    df = impute_column(df, "age_60_and_above", predictors, verbose)
    df = impute_column(df, "gender", predictors + ["age_60_and_above"], verbose)
    return df


def to_integer_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the raw date and convert every remaining column to integers."""
    return df.drop(columns=["test_date"]).astype(int)


def preprocess(train_raw: pd.DataFrame, test_raw: pd.DataFrame, verbose: bool = True):
    """Run the full preprocessing pipeline and return integer train/test frames."""
    train = clean_and_encode(train_raw)
    test = clean_and_encode(test_raw)
    train, test = add_test_date_numeric(train, test)
    if verbose:
        print("Imputing missing demographic values (test file)")
    test = impute_demographics(test, verbose)
    if verbose:
        print("Imputing missing demographic values (training file)")
    train = impute_demographics(train, verbose)
    return to_integer_frame(train), to_integer_frame(test)


def class_distribution(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Counts and percentages of each corona_result class in the raw data (Table 1)."""
    labels = df_raw[config.TARGET].astype(str).str.strip().str.lower()
    counts = labels.value_counts()
    return pd.DataFrame({"Count": counts, "Percentage": (100 * counts / counts.sum()).round(2)})
