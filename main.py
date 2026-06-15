from features.match_features import generate_match_features
from features.country_profiles import generate_wc_country_profiles
from features.dataset_builder import build_training_dataset
from models.train_model import train_model

def main():
    country_df = generate_wc_country_profiles(verbose=False)
    country_df["country"] = country_df["country"].astype(str).str.strip().str.lower()
    wc_teams = set(country_df["country"])
    
    matches_df, elo_df = generate_match_features(
        wc_teams=None,  
        verbose=False
    )

    matches_df["home_team"] = matches_df["home_team"].str.lower()
    matches_df["away_team"] = matches_df["away_team"].str.lower()

    matches_df = matches_df.merge(
        country_df.add_prefix("home_"),
        left_on="home_team",
        right_on="home_country",
        how="left"
    ).drop(columns=["home_country"]) 

    matches_df = matches_df.merge(
        country_df.add_prefix("away_"),
        left_on="away_team",
        right_on="away_country",
        how="left"
    ).drop(columns=["away_country"]) 

    matches_df = matches_df[
        matches_df["home_team"].isin(wc_teams) & 
        matches_df["away_team"].isin(wc_teams)
    ].copy()

    if matches_df.empty:
        raise ValueError("❌ matches_df is empty! Check that team names match exactly between your datasets.")

    matches_df["elo_diff_scaled"] = matches_df["elo_diff"] / 400
    
    matches_df["host_adjusted_elo_diff"] = matches_df["elo_diff_scaled"] + (
        (matches_df["home_is_host"] * 0.3) - (matches_df["away_is_host"] * 0.3)
    )
    
    matches_df["attack_defense_gap"] = (
        matches_df["home_avg_goals_scored"] - matches_df["away_avg_goals_conceded"]
    )
    matches_df["goals_scored_diff"] = matches_df["home_avg_goals_scored"] - matches_df["away_avg_goals_scored"]
    matches_df["goal_diff_form"] = (matches_df["goals_scored_diff"] + matches_df["form_diff"])
    
    matches_df["elo_form_interaction"] = (matches_df["host_adjusted_elo_diff"] * matches_df["form_diff"])
    matches_df["elo_tournament_pressure"] = (matches_df["host_adjusted_elo_diff"] * matches_df["tournament_strength"])

    features = [
        "host_adjusted_elo_diff",
        "elo_tournament_pressure",
        "elo_form_interaction",
        "goal_diff_form",
        "form_diff",
        "continent_advantage",
        "goals_scored_diff",
        "attack_defense_gap",
        "tournament_strength",
        "is_neutral"
    ]
    
    train_model(matches_df, features)

    final_df = build_training_dataset(matches_df, country_df)
    final_df.to_csv("data/final_training_data.csv", index=False)

if __name__ == "__main__":
    main()