import json

import pytest

import pandas as pd

from opticar import cli
from opticar import config as C
from opticar import predict as P
from opticar.training import train


@pytest.fixture(scope="module")
def model_path(tmp_path_factory, raw_df):
    d = tmp_path_factory.mktemp("opticar")
    csv = d / "raw.csv"
    raw_df.to_csv(csv, index=False)
    train(csv, d / "m.joblib", d / "reports", cv=2, plots=False, quiet=True)
    return d / "m.joblib"


def test_train_writes_artifact_and_metrics(model_path):
    assert model_path.exists()
    metrics = json.loads((model_path.parent / "reports" / "metrics.json").read_text())
    assert metrics["best_model"] != "Baseline (mean)"


def test_predict_single(model_path, capsys):
    rc = cli.main(["predict", "--model-path", str(model_path), "--brand", "bmw", "--model", "3er",
                   "--age", "8", "--power", "150", "--km", "100000", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out["low_eur"] < out["predicted_price_eur"] < out["high_eur"]


def test_predict_requires_numeric_inputs(model_path, capsys):
    assert cli.main(["predict", "--model-path", str(model_path), "--brand", "bmw"]) == 2


def test_predict_batch(model_path, tmp_path):
    src, dst = tmp_path / "in.csv", tmp_path / "out.csv"
    src.write_text("brand,model,Age,powerPS,kilometer\nbmw,3er,5,150,50000\naudi,a4,12,120,150000\n")
    assert cli.main(["predict", "--model-path", str(model_path), "--batch", str(src), "--output", str(dst)]) == 0
    assert "predicted_price_eur" in dst.read_text().splitlines()[0]


def test_missing_model_gives_clean_error(tmp_path, capsys):
    assert cli.main(["info", "--model-path", str(tmp_path / "none.joblib")]) == 1
    assert "opticar train" in capsys.readouterr().err


def test_omitted_optional_fields_equal_the_imputed_defaults(model_path):
    """Regression: None must be treated as missing (imputed), not as a new category."""
    art = P.load_artifact(model_path)
    imputer = art["pipeline"].regressor_.named_steps["prep"].named_transformers_["cat"].named_steps["impute"]
    modes = dict(zip(C.CATEGORICAL, imputer.statistics_))
    base = {"brand": "bmw", "model": "3er", "Age": 8.0, "powerPS": 150.0, "kilometer": 100000.0}
    omitted = P.predict_frame(pd.DataFrame([{**base, **{c: None for c in modes if c not in base}}]), art)
    explicit = P.predict_frame(pd.DataFrame([{**modes, **base}]), art)
    assert omitted["predicted_price_eur"].iloc[0] == explicit["predicted_price_eur"].iloc[0]
