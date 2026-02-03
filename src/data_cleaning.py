import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """Load raw dataset"""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw car dataset"""

    # Drop irrelevant columns
    drop_cols = ['name', 'dateCrawled', 'dateCreated', 'postalCode', 'lastSeen']
    df = df.drop(columns=drop_cols, errors='ignore')

    # Remove duplicates
    df = df.drop_duplicates()

    # Filter realistic ranges
    df = df[
        (df['yearOfRegistration'].between(1950, 2018)) &
        (df['price'].between(100, 150000)) &
        (df['powerPS'].between(10, 500))
    ]

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values"""

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    return df


def save_data(df: pd.DataFrame, path: str) -> None:
    """Save cleaned dataset"""
    df.to_csv(path, index=False)


if __name__ == "__main__":
    df = load_data("../data/raw/cars_sampled.csv")
    df = clean_data(df)
    df = handle_missing_values(df)
    save_data(df, "../data/processed/cars_cleaned.csv")
