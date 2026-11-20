import pandas as pd
import pytest

from opticar import config as C
from opticar import data as D


def test_clean_filters_outliers_and_drops_noise_columns(raw_df):
    bad = raw_df.iloc[:3].copy()
    bad["price"] = [50, 999_999, 5000]            # too cheap, too expensive, fine
    bad["yearOfRegistration"] = [2005, 2005, 2018]  # third is after the crawl date
    out = D.clean(pd.concat([raw_df.iloc[3:], bad], ignore_index=True))
    assert out["price"].between(*C.PRICE_RANGE).all()
    assert out["yearOfRegistration"].max() <= C.REFERENCE_YEAR
    assert not {"abtest", "name", "postalCode", "seller"} & set(out.columns)


def test_clean_removes_duplicates(raw_df):
    doubled = pd.concat([raw_df, raw_df], ignore_index=True)
    assert len(D.clean(doubled)) == len(D.clean(raw_df))


def test_age_later_month_is_younger_and_invalid_month_is_midyear():
    df = pd.DataFrame({"yearOfRegistration": [2006, 2006, 2006],
                       "monthOfRegistration": [1, 12, 0]})
    age = D.add_age(df)["Age"].tolist()
    assert age[0] > age[1]                       # January 2006 is older than December 2006
    assert age[2] == pytest.approx(2016.25 - (2006 + 5.5 / 12), abs=0.01)  # month 0 -> June
    assert "yearOfRegistration" not in D.add_age(df).columns


def test_age_never_negative():
    df = pd.DataFrame({"yearOfRegistration": [2016], "monthOfRegistration": [12]})
    assert D.add_age(df)["Age"].iloc[0] == 0


def test_prepare_keeps_missing_values_for_the_pipeline(raw_df):
    out = D.prepare(raw_df)
    assert out["notRepairedDamage"].isna().any()  # imputation happens in the pipeline
    assert set(C.FEATURES) <= set(out.columns)
