"""Central configuration: paths, column groups and cleaning bounds."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DATA = ROOT / "data" / "raw" / "cars_sampled.csv"
MODEL_PATH = ROOT / "models" / "opticar.joblib"
REPORTS_DIR = ROOT / "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Listings were crawled in March-April 2016, so "age" is measured against
# that point in time (decimal year) rather than today.
REFERENCE_DATE = 2016.25
REFERENCE_YEAR = 2016  # no car can be registered after the crawl year

TARGET = "price"

# Identifiers / timestamps / constant-or-noise columns carry no pricing signal.
# (`abtest` is an eBay experiment flag, `seller`/`offerType` are ~constant.)
DROP_COLS = [
    "name", "dateCrawled", "dateCreated", "postalCode", "lastSeen",
    "abtest", "seller", "offerType",
]

NUMERIC = ["powerPS", "kilometer", "Age"]
CATEGORICAL = ["vehicleType", "gearbox", "model", "fuelType", "brand", "notRepairedDamage"]
FEATURES = NUMERIC + CATEGORICAL

# Domain-driven bounds used to filter implausible listings.
YEAR_RANGE = (1950, REFERENCE_YEAR)
PRICE_RANGE = (100, 150_000)
POWER_RANGE = (10, 500)
