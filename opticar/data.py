"""Loading, cleaning and feature engineering for the raw listings."""
from __future__ import annotations

import pandas as pd

from opticar import config as C


def load_raw(path=C.RAW_DATA) -> pd.DataFrame:
    return pd.read_csv(path)


def add_age(df: pd.DataFrame, reference_date: float = C.REFERENCE_DATE) -> pd.DataFrame:
    """Vehicle age in years at listing time, from registration year + month.

    Registration is placed at the middle of its month, so later months in the
    same year are *younger*. A month of 0 is invalid in the source data (~10%
    of rows) and is treated as mid-year (6) rather than silently skewing age.
    """
    df = df.copy()
    month = df["monthOfRegistration"].where(df["monthOfRegistration"].between(1, 12), 6)
    registered = df["yearOfRegistration"] + (month - 0.5) / 12
    df["Age"] = (reference_date - registered).clip(lower=0).round(2)
    return df.drop(columns=["yearOfRegistration", "monthOfRegistration"])


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level cleaning that involves no fitted statistics (so no leakage).

    Imputation of missing values is deliberately *not* done here; it lives in
    the sklearn pipeline and is fit on the training split only.
    """
    df = df.drop(columns=C.DROP_COLS, errors="ignore").drop_duplicates()
    df = df[
        df["yearOfRegistration"].between(*C.YEAR_RANGE)
        & df[C.TARGET].between(*C.PRICE_RANGE)
        & df["powerPS"].between(*C.POWER_RANGE)
    ]
    return df.reset_index(drop=True)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Raw listings -> modelling frame (cleaned, Age engineered)."""
    return add_age(clean(df))
