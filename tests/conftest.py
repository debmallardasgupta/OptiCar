import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def raw_df():
    """Small synthetic frame in the raw-listing schema (no dataset needed)."""
    rng = np.random.default_rng(0)
    n = 400
    brands = rng.choice(["bmw", "audi", "vw"], n)
    year = rng.integers(1995, 2016, n)
    power = rng.integers(60, 300, n)
    km = rng.choice([5000, 50000, 100000, 150000], n)
    price = (2000 + (year - 1990) * 300 + power * 25 - km * 0.02
             + (brands == "bmw") * 1500 + rng.normal(0, 300, n)).clip(150)
    df = pd.DataFrame({
        "dateCrawled": "2016-03-30", "name": "x", "seller": "private",
        "offerType": "offer", "abtest": rng.choice(["test", "control"], n),
        "price": price.round(), "vehicleType": rng.choice(["limousine", "suv"], n),
        "yearOfRegistration": year, "gearbox": rng.choice(["manual", "automatic"], n),
        "powerPS": power, "model": np.where(brands == "bmw", "3er", "golf"),
        "kilometer": km, "monthOfRegistration": rng.integers(0, 13, n),
        "fuelType": rng.choice(["petrol", "diesel"], n), "brand": brands,
        "notRepairedDamage": rng.choice(["yes", "no", None], n),
        "dateCreated": "2016-03-30", "postalCode": 12345, "lastSeen": "2016-04-01",
    })
    return df
