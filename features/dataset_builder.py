import pandas as pd


def build_training_dataset(matches_df, country_df, verbose=False):

    home_df = country_df.add_prefix("home_")
    away_df = country_df.add_prefix("away_")

    df = matches_df.merge(
        home_df,
        left_on="home_team",
        right_on="home_country",
        how="left"
    )

    df = df.merge(
        away_df,
        left_on="away_team",
        right_on="away_country",
        how="left"
    )

    # Drop duplicate columns
    df.drop(columns=["home_country", "away_country"], inplace=True, errors="ignore")

    if verbose:
        print("\n=== FINAL TRAINING DATASET ===")
        print(df.head())
        print("\nShape:", df.shape)
        print("\nMissing values:\n", df.isnull().sum())

    return df