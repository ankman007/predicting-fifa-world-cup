from features.match_features import generate_match_features
from features.country_profiles import generate_wc_country_profiles
from features.dataset_builder import build_training_dataset
from models.train_model import train_model


def main():
    country_df = generate_wc_country_profiles(verbose=False)
    wc_teams = set(country_df["country"])
    country = country_df.copy()
    
    matches_df, elo_df = generate_match_features(wc_teams=wc_teams, verbose=False)
    
    matches_df = matches_df.merge(
        country_df.add_prefix("home_"),
        left_on="home_team",
        right_on="home_country",
        how="left"
    )
    matches_df = matches_df.merge(
        country_df.add_prefix("away_"),
        left_on="away_team",
        right_on="away_country",
        how="left"
    )
    
    matches_df["elo_form_interaction"] = matches_df["elo_diff"] * matches_df["form_diff"]

    matches_df["attack_defense_gap"] = (
        matches_df["home_avg_goals_scored"] - matches_df["away_avg_goals_conceded"]
    )
    
    matches_df["elo_tournament_pressure"] = matches_df["elo_diff"] * matches_df["tournament_strength"]
    matches_df["form_tournament_pressure"] = matches_df["form_diff"] * matches_df["tournament_strength"]

    matches_df["home_advantage"] = 1
    matches_df["elo_diff_scaled"] = matches_df["elo_diff"] / 400
    matches_df["attack_strength"] = matches_df["home_avg_goals_scored"] - matches_df["away_avg_goals_conceded"]
    
    features = [
        "home_elo",
        "away_elo",
        "elo_diff",
        "home_form",
        "away_form",
        "form_diff",
        "home_avg_goals_scored",
        "away_avg_goals_scored",
        "goals_scored_diff",
        "home_avg_goals_conceded",
        "away_avg_goals_conceded",
        "goals_conceded_diff",
        "elo_form_interaction",
        "attack_defense_gap",
        "tournament_strength",
        "elo_tournament_pressure",
        "form_tournament_pressure"
    ]
    
    train_model(matches_df, features)
    
    final_df = build_training_dataset(matches_df, country_df,)
    final_df.to_csv("data/final_training_data.csv", index=False)


if __name__ == "__main__":
    main()