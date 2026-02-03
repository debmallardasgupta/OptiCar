import pandas as pd


def create_age_feature(df: pd.DataFrame, current_year: int = 2018) -> pd.DataFrame:
    """Create vehicle age feature"""

    df['monthOfRegistration'] = df['monthOfRegistration'] / 12
    df['Age'] = (current_year - df['yearOfRegistration']) + df['monthOfRegistration']
    df['Age'] = df['Age'].round(2)

    df = df.drop(columns=['yearOfRegistration', 'monthOfRegistration'])
    return df


if __name__ == "__main__":
    df = pd.read_csv("../data/processed/cars_cleaned.csv")
    df = create_age_feature(df)
    df.to_csv("../data/processed/cars_featured.csv", index=False)