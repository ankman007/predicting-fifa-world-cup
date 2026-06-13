import pandas as pd
from utils import assign_climate_points

def generate_wc_country_profiles(
    wc_groups_path="data/world_cup_groups.csv",
    confed_scaling_path="data/continental_confederations_scaling.csv",
    temp_path="data/country_10yr_temp_averages.csv",
    fifa_ranking_path="data/fifa_world_ranking.csv",
    verbose=True
):
    # 1. Load Datasets
    wc_groups = pd.read_csv(wc_groups_path)
    confed_scaling = pd.read_csv(confed_scaling_path)
    temp_data = pd.read_csv(temp_path)
    fifa_ranking = pd.read_csv(fifa_ranking_path)

    # 2. Merge Confederation Scaling Factors
    country_profile = pd.merge(
        wc_groups,
        confed_scaling[["confederation", "qualifying_difficulty"]], 
        on="confederation",
        how="left"
    )

    # 3. Merge Climate Data & Calculate Bonus Points
    country_profile = pd.merge(
        country_profile, 
        temp_data[["country", "mean_temp_last_10_yrs"]], 
        on="country", 
        how="left"
    )
    country_profile['climate_bonus_points'] = (
        country_profile['mean_temp_last_10_yrs'].apply(assign_climate_points)
    )

    # 4. Process and Deduplicate FIFA Rankings (Get latest per country/acronym)
    ranking_cols = ["date", "semester", "rank", "acronym", "total.points", "previous.points", "diff.points"]
    
    latest_rankings = (
        fifa_ranking.sort_values(by=["date", "semester"], ascending=[False, False])
        .drop_duplicates(subset=["acronym"], keep="first")
        .drop_duplicates(subset=["country"], keep="first")
        [ranking_cols]
    )

    # 5. Merge FIFA Rankings
    country_profile = pd.merge(country_profile, latest_rankings, on="acronym", how="left")

    # 6. Optional Logging & Integrity Checks
    if verbose:
        print("Country Profile:\n", country_profile)
        print(f"\nDataFrame Dimensions (Rows, Columns): {country_profile.shape}")
        print("\n--- Detailed Column Breakdown ---")
        country_profile.info()
        
        missing_teams = wc_groups[~wc_groups['country'].isin(country_profile['country'].dropna())]
        if not missing_teams.empty:
            print("\n⚠️ WARNING: The following World Cup teams were dropped or have missing data:")
            print(missing_teams['country'].tolist())

    return country_profile

df_profiles = generate_wc_country_profiles(verbose=True)