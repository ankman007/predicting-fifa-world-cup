from features.match_features import generate_match_features
from features.country_profiles import generate_wc_country_profiles


def main():

    matches_df, elo_df = generate_match_features(verbose=True)

    country_profiles_df = generate_wc_country_profiles(verbose=True)

    print("\n=== Match Feature Dataset ===")
    print(matches_df.head())

    print("\n=== Country Profiles Ready ===")
    print(country_profiles_df.head())


if __name__ == "__main__":
    main()