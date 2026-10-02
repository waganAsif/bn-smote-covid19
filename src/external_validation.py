"""Loading the independent (CMC Hospital) dataset for external validation."""

from pathlib import Path

import pandas as pd

from . import config

YES_NO = {"yes": 1, "no": 0, "1": 1, "0": 0, "positive": 1, "negative": 0}


def _encode(series: pd.Series) -> pd.Series:
    encoded = series.astype(str).str.strip().str.lower().map(YES_NO)
    if encoded.isna().any():
        bad = sorted(series[encoded.isna()].astype(str).unique())[:5]
        raise ValueError(f"Column '{series.name}' contains values that are not Yes/No or 1/0: {bad}")
    return encoded.astype(int)


def load_cmc(path: Path = config.CMC_FILE) -> pd.DataFrame:
    """Load data/cmc.csv and return the five symptoms plus corona_result as 0/1 integers.

    The file may use the original column names (Breathing Problem, Fever, Dry
    Cough, Sore throat, Headache, COVID-19) with Yes/No values, or the
    training-data names (shortness_of_breath, fever, cough, sore_throat,
    head_ache, corona_result) with 0/1 values. Other columns are ignored.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} was not found. Add the independent dataset as data/cmc.csv "
            "(see data/README.md for the expected columns)."
        )
    raw = pd.read_csv(path)
    raw.columns = [c.strip() for c in raw.columns]
    renamed = raw.rename(columns=config.CMC_COLUMN_MAP)
    needed = config.EXTERNAL_PARENTS + [config.TARGET]
    missing = [c for c in needed if c not in renamed.columns]
    if missing:
        raise ValueError(f"{path.name} is missing the columns: {missing}")
    data = renamed[needed].dropna()
    return data.apply(_encode)
