import pandas as pd
from collections import defaultdict


def get_recent_average(history, team, window=5):
    recent = history[team][-window:]

    if len(recent) == 0:
        return 0

    return sum(recent) / len(recent)


def get_match_target(home_score, away_score):
    if home_score > away_score:
        return 0  # Home Win

    if home_score < away_score:
        return 2  # Away Win

    return 1  # Draw


def generate_match_features(
    matches_path="data/football_match_results.csv",
    elo_k=40,
    form_window=5,
    verbose=True,
):
    matches = pd.read_csv(matches_path)

    matches["date"] = pd.to_datetime(matches["date"])

    matches = (
        matches
        .dropna(subset=["home_score", "away_score"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Historical tracking
    team_history = defaultdict(list)
    goals_scored_history = defaultdict(list)
    goals_conceded_history = defaultdict(list)

    # Feature storage
    home_form_list = []
    away_form_list = []

    home_avg_goals_scored = []
    away_avg_goals_scored = []

    home_avg_goals_conceded = []
    away_avg_goals_conceded = []

    home_elo_list = []
    away_elo_list = []

    # Initialize Elo
    teams = set(matches["home_team"]).union(matches["away_team"])

    elo = {team: 1500 for team in teams}

    for _, match in matches.iterrows():

        home = match["home_team"]
        away = match["away_team"]

        # -------------------------
        # Recent Goal Scoring Form
        # -------------------------

        home_avg_goals_scored.append(
            get_recent_average(
                goals_scored_history,
                home,
                form_window
            )
        )

        away_avg_goals_scored.append(
            get_recent_average(
                goals_scored_history,
                away,
                form_window
            )
        )

        # -------------------------
        # Recent Goal Conceding Form
        # -------------------------

        home_avg_goals_conceded.append(
            get_recent_average(
                goals_conceded_history,
                home,
                form_window
            )
        )

        away_avg_goals_conceded.append(
            get_recent_average(
                goals_conceded_history,
                away,
                form_window
            )
        )

        # -------------------------
        # Recent Match Form
        # -------------------------

        home_form_list.append(
            sum(team_history[home][-form_window:])
        )

        away_form_list.append(
            sum(team_history[away][-form_window:])
        )

        # -------------------------
        # Elo Before Match
        # -------------------------

        home_elo = elo[home]
        away_elo = elo[away]

        home_elo_list.append(home_elo)
        away_elo_list.append(away_elo)

        expected_home = (
            1 /
            (1 + 10 ** ((away_elo - home_elo) / 400))
        )

        expected_away = 1 - expected_home

        # -------------------------
        # Match Result
        # -------------------------

        if match["home_score"] > match["away_score"]:
            actual_home = 1
            actual_away = 0

            home_points = 3
            away_points = 0

        elif match["home_score"] < match["away_score"]:
            actual_home = 0
            actual_away = 1

            home_points = 0
            away_points = 3

        else:
            actual_home = 0.5
            actual_away = 0.5

            home_points = 1
            away_points = 1

        # -------------------------
        # Update Histories
        # -------------------------

        team_history[home].append(home_points)
        team_history[away].append(away_points)

        goals_scored_history[home].append(
            match["home_score"]
        )
        goals_scored_history[away].append(
            match["away_score"]
        )

        goals_conceded_history[home].append(
            match["away_score"]
        )
        goals_conceded_history[away].append(
            match["home_score"]
        )

        # -------------------------
        # Update Elo
        # -------------------------

        elo[home] = (
            home_elo +
            elo_k * (actual_home - expected_home)
        )

        elo[away] = (
            away_elo +
            elo_k * (actual_away - expected_away)
        )

    # ==================================================
    # Create Features
    # ==================================================

    matches["home_form"] = home_form_list
    matches["away_form"] = away_form_list
    matches["form_diff"] = (
        matches["home_form"] -
        matches["away_form"]
    )

    matches["home_avg_goals_scored"] = home_avg_goals_scored
    matches["away_avg_goals_scored"] = away_avg_goals_scored
    matches["goals_scored_diff"] = (
        matches["home_avg_goals_scored"] -
        matches["away_avg_goals_scored"]
    )

    matches["home_avg_goals_conceded"] = home_avg_goals_conceded
    matches["away_avg_goals_conceded"] = away_avg_goals_conceded
    matches["goals_conceded_diff"] = (
        matches["away_avg_goals_conceded"] -
        matches["home_avg_goals_conceded"]
    )

    matches["home_elo"] = home_elo_list
    matches["away_elo"] = away_elo_list
    matches["elo_diff"] = (
        matches["home_elo"] -
        matches["away_elo"]
    )

    matches["target"] = matches.apply(
        lambda row: get_match_target(
            row["home_score"],
            row["away_score"]
        ),
        axis=1
    )

    elo_df = pd.DataFrame(
        elo.items(),
        columns=["team", "elo"]
    )
    
    if verbose:
        print("\n=== Match Feature Dataset ===")
        print(matches.head())

        print(f"\nDataset Shape: {matches.shape}")

        print("\n=== Target Distribution ===")
        print(matches["target"].value_counts())

        print("\n=== Missing Values ===")
        print(matches.isna().sum())

        print("\n=== Top 20 Elo Teams ===")
        print(
            elo_df.sort_values(
                "elo",
                ascending=False
            ).head(20)
        )

    return matches, elo_df

