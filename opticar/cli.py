"""Command-line interface: `opticar train | predict | info`."""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from opticar import __version__
from opticar import config as C
from opticar import predict as P


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="opticar", description="Used-car price prediction (scikit-learn pipeline).")
    p.add_argument("--version", action="version", version=f"opticar {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="Clean data, compare models, save the best pipeline.")
    t.add_argument("--data", default=str(C.RAW_DATA), help="Raw listings CSV.")
    t.add_argument("--model-path", default=str(C.MODEL_PATH), help="Where to save the artifact.")
    t.add_argument("--reports-dir", default=str(C.REPORTS_DIR), help="Where to write metrics/plots.")
    t.add_argument("--cv", type=int, default=3, help="CV folds for model selection (default 3).")
    t.add_argument("--tune", action="store_true", help="Randomised hyper-parameter search on the winner.")
    t.add_argument("--no-plots", action="store_true", help="Skip PNG plots.")

    pr = sub.add_parser("predict", help="Estimate the price of one car, or a CSV of cars.")
    pr.add_argument("--model-path", default=str(C.MODEL_PATH))
    pr.add_argument("--brand"); pr.add_argument("--model")
    pr.add_argument("--age", type=float, help="Age in years at listing time.")
    pr.add_argument("--power", type=float, dest="powerPS", help="Engine power in PS.")
    pr.add_argument("--km", type=float, dest="kilometer", help="Odometer reading (km).")
    pr.add_argument("--fuel", dest="fuelType"); pr.add_argument("--gearbox")
    pr.add_argument("--type", dest="vehicleType", help="e.g. limousine, suv, small car.")
    pr.add_argument("--damaged", dest="notRepairedDamage", choices=["yes", "no"],
                    help="Unrepaired damage? (default: imputed)")
    pr.add_argument("--batch", metavar="CSV", help="Predict for every row of a CSV instead.")
    pr.add_argument("--output", metavar="CSV", help="With --batch: write results here.")
    pr.add_argument("--json", action="store_true", help="Machine-readable output.")

    i = sub.add_parser("info", help="Show model metrics and valid input values.")
    i.add_argument("--model-path", default=str(C.MODEL_PATH))
    i.add_argument("--brand", help="List the models seen for this brand.")
    return p


def _cmd_train(a) -> int:
    from opticar.training import train
    s = train(a.data, a.model_path, a.reports_dir, cv=a.cv, tune=a.tune, plots=not a.no_plots)
    m = s["test_metrics"][s["best_model"]]
    print(f"\nBest model: {s['best_model']}")
    print(f"Test  RMSE: EUR {m['RMSE_eur']:,.0f} | MAE: EUR {m['MAE_eur']:,.0f} | "
          f"R2: {m['R2_eur']:.3f} (log-price R2: {m['R2_log']:.3f})")
    print(f"Saved model -> {s['model_path']}  ({s['seconds']}s)")
    return 0


def _cmd_predict(a) -> int:
    art = P.load_artifact(a.model_path)
    if a.batch:
        df = pd.read_csv(a.batch)
        out = P.predict_frame(df, art)
        if a.output:
            out.to_csv(a.output, index=False)
            print(f"Wrote {len(out)} predictions -> {a.output}")
        else:
            print(out.to_string(index=False) if not a.json else out.to_json(orient="records", indent=2))
        return 0

    row = {c: getattr(a, c) for c in C.FEATURES if hasattr(a, c)}
    row["Age"] = a.age
    if any(row[c] is None for c in C.NUMERIC):
        print("error: --age, --power and --km are required (or use --batch).", file=sys.stderr)
        return 2
    for w in P.validate({k: v for k, v in row.items() if v is not None}, art):
        print(f"warning: {w}", file=sys.stderr)
    res = P.predict_frame(pd.DataFrame([row]), art).iloc[0]
    if a.json:
        print(json.dumps({"predicted_price_eur": res.predicted_price_eur,
                          "low_eur": res.low_eur, "high_eur": res.high_eur}))
    else:
        print(f"Estimated price: EUR {res.predicted_price_eur:,.0f}")
        print(f"Likely range (80% of held-out cases): EUR {res.low_eur:,.0f} - {res.high_eur:,.0f}")
        print("Note: trained on 2016 listings; prices reflect that market.")
    return 0


def _cmd_info(a) -> int:
    art = P.load_artifact(a.model_path)
    if a.brand:
        models = art["models_by_brand"].get(a.brand.lower())
        if not models:
            print(f"Unknown brand {a.brand!r}. Brands: {', '.join(sorted(art['models_by_brand']))}")
            return 1
        print(f"{a.brand.lower()}: {', '.join(models)}")
        return 0
    s = art["summary"]
    m = s["test_metrics"][s["best_model"]]
    print(f"Model: {s['best_model']}  (trained on {s['n_train']:,} / tested on {s['n_test']:,} listings)")
    print(f"Test RMSE EUR {m['RMSE_eur']:,.0f} | MAE EUR {m['MAE_eur']:,.0f} | R2 {m['R2_eur']:.3f}")
    for col, vals in art["categories"].items():
        print(f"{col}: {', '.join(map(str, vals))}")
    print(f"brands: {', '.join(sorted(art['models_by_brand']))}")
    return 0


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return {"train": _cmd_train, "predict": _cmd_predict, "info": _cmd_info}[args.command](args)
    except P.ModelNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
