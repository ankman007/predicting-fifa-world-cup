import pandas as pd
from collections import defaultdict
from constants import TOURNAMENT_STRENGTH

# 2026 World Cup Hosts
WC_2026_HOSTS = {"united states", "mexico", "canada", "usa"}

# Comprehensive dictionary for continent mapping to calculate geographic adaptation
CONTINENT_MAP = {
    # Europe
    "germany": "EU", "france": "EU", "england": "EU", "italy": "EU", "spain": "EU", 
    "netherlands": "EU", "belgium": "EU", "portugal": "EU", "croatia": "EU", "scotland": "EU",
    # South America
    "brazil": "SA", "argentina": "SA", "uruguay": "SA", "colombia": "SA", "chile": "SA",
    # North America
    "united states": "NA", "usa": "NA", "mexico": "NA", "canada": "NA", "costa rica": "NA",
    # Africa
    "morocco": "AF", "senegal": "AF", "tunisia": "AF", "cameroon": "AF", "ghana": "AF",
    # Asia / Oceania
    "japan": "AS", "south korea": "AS", "australia": "AS", "saudi arabia": "AS", "iran": "AS"
}

def get_recent_average(history, team, window=5):
    recent = history[team][-window:]
    return sum(recent) / len(recent) if recent else 0

def generate_match_features(
    matches_path="data/football_match_results.csv",
    elo_k=40,
    form_window=5,
    wc_teams=None,
    verbose=True,
):
    # =========================
    # LOAD DATA
    # =========================
    matches = pd.read_csv(matches_path)
    matches["date"] = pd.to_datetime(matches["date"])

    # Normalize strings
    matches["tournament"] = matches["tournament"].astype(str).str.strip().str.lower()
    matches["home_team"] = matches["home_team"].astype(str).str.strip().str.lower()
    matches["away_team"] = matches["away_team"].astype(str).str.strip().str.lower()
    matches["country"] = matches["country"].astype(str).str.strip().str.lower()

    # =========================
    # TOURNAMENT STRENGTH
    # =========================
    matches["tournament_strength"] = matches["tournament"].map(
        lambda x: TOURNAMENT_STRENGTH.get(x, 0.5)
    )

    # =========================
    # FILTERING (Truncating Ancient Data to Focus on Modern Era Mechanics)
    # =========================
    # Filtering the start date to 2000-01-01 prevents historic data from throwing off modern Elo values
    matches = matches[matches["date"] >= "2000-01-01"].copy()

    matches = (
        matches
        .dropna(subset=["home_score", "away_score"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    if wc_teams is not None:
        wc_teams_lower = {t.lower() for t in wc_teams}
        matches = matches[
            matches["home_team"].isin(wc_teams_lower) &
            matches["away_team"].isin(wc_teams_lower)
        ].reset_index(drop=True)

    # =========================
    # HISTORIES
    # =========================
    team_history = defaultdict(list)
    goals_scored_history = defaultdict(list)
    goals_conceded_history = defaultdict(list)

    # =========================
    # FEATURE STORAGE
    # =========================
    home_form_list = []
    away_form_list = []
    home_avg_goals_scored = []
    away_avg_goals_scored = []
    home_avg_goals_conceded = []
    away_avg_goals_conceded = []
    home_elo_list = []
    away_elo_list = []

    is_neutral_list = []
    home_is_host_list = []
    away_is_host_list = []
    same_continent_list = []
    binary_target_list = []

    # =========================
    # INITIAL ELO
    # =========================
    teams = set(matches["home_team"]).union(matches["away_team"])
    elo = {team: 1500 for team in teams}

    # =========================
    # MAIN LOOP
    # =========================
    for _, match in matches.iterrows():
        home = match["home_team"]
        away = match["away_team"]
        match_country = match["country"]

        # -------------------------
        # VENUE CONTEXT & GEOGRAPHY
        # -------------------------
        is_neutral_val = 1 if ("neutral" in matches.columns and match["neutral"] is True) or (match_country != home) else 0
        is_neutral_list.append(is_neutral_val)

        home_is_host_val = 1 if home in WC_2026_HOSTS or home == match_country else 0
        away_is_host_val = 1 if away in WC_2026_HOSTS or away == match_country else 0
        home_is_host_list.append(home_is_host_val)
        away_is_host_list.append(away_is_host_val)

        home_cont = CONTINENT_MAP.get(home, "UNKNOWN_H")
        away_cont = CONTINENT_MAP.get(away, "UNKNOWN_A")
        match_cont = CONTINENT_MAP.get(match_country, "UNKNOWN_M")
        
        if home_cont == match_cont and away_cont != match_cont:
            same_continent_list.append(1)
        elif away_cont == match_cont and home_cont != match_cont:
            same_continent_list.append(-1)
        else:
            same_continent_list.append(0)

        # -------------------------
        # FORM & GOAL HISTORIES
        # -------------------------
        home_form_list.append(sum(team_history[home][-form_window:]))
        away_form_list.append(sum(team_history[away][-form_window:]))

        home_avg_goals_scored.append(get_recent_average(goals_scored_history, home, form_window))
        away_avg_goals_scored.append(get_recent_average(goals_scored_history, away, form_window))

        home_avg_goals_conceded.append(get_recent_average(goals_conceded_history, home, form_window))
        away_avg_goals_conceded.append(get_recent_average(goals_conceded_history, away, form_window))

        # -------------------------
        # ADAPTIVE ELO SYSTEM
        # -------------------------
        home_elo = elo[home]
        away_elo = elo[away]
        home_elo_list.append(home_elo)
        away_elo_list.append(away_elo)

        # Apply a +100 point home advantage calculation strictly for non-neutral games
        home_advantage_bump = 100 if is_neutral_val == 0 else 0
        expected_home = 1 / (1 + 10 ** (((away_elo) - (home_elo + home_advantage_bump)) / 400))
        expected_away = 1 - expected_home
        
        # -------------------------
        # BINARY TARGET MECHANICS (1: Home Wins/Advances, 0: Away Wins/Advances)
        # -------------------------
        if match["home_score"] > match["away_score"]:
            actual_home, actual_away = 1, 0
            home_points, away_points = 3, 0
            binary_target_list.append(1)
        elif match["home_score"] < match["away_score"]:
            actual_home, actual_away = 0, 1
            home_points, away_points = 0, 3
            binary_target_list.append(0)
        else:
            actual_home = actual_away = 0.5
            home_points = away_points = 1
            # Resolve historical ties by favoring the pre-match Elo favorite (Shootout Simulation)
            resolved_draw = 1 if home_elo >= away_elo else 0
            binary_target_list.append(resolved_draw)

        # Update History
        team_history[home].append(home_points)
        team_history[away].append(away_points)
        goals_scored_history[home].append(match["home_score"])
        goals_scored_history[away].append(match["away_score"])
        goals_conceded_history[home].append(match["away_score"])
        goals_conceded_history[away].append(match["home_score"])

        # Update dynamic Elo map
        dynamic_k = elo_k * (0.5 + match["tournament_strength"])
        elo[home] += dynamic_k * (actual_home - expected_home)
        elo[away] += dynamic_k * (actual_away - expected_away)

    # =========================
    # BUILD DATAFRAME EXTENSIONS
    # =========================
    matches["is_neutral"] = is_neutral_list
    matches["home_is_host"] = home_is_host_list
    matches["away_is_host"] = away_is_host_list
    matches["continent_advantage"] = same_continent_list
    matches["target"] = binary_target_list

    matches["home_form"] = home_form_list
    matches["away_form"] = away_form_list
    matches["form_diff"] = matches["home_form"] - matches["away_form"]
    matches["home_avg_goals_scored"] = home_avg_goals_scored
    matches["away_avg_goals_scored"] = away_avg_goals_scored
    matches["home_avg_goals_conceded"] = home_avg_goals_conceded
    matches["away_avg_goals_conceded"] = away_avg_goals_conceded
    
    matches["goals_conceded_diff"] = matches["away_avg_goals_conceded"] - matches["home_avg_goals_conceded"]
    matches["home_elo"] = home_elo_list
    matches["away_elo"] = away_elo_list
    matches["elo_diff"] = matches["home_elo"] - matches["away_elo"]
    matches["elo_prob_home"] = 1 / (1 + 10 ** (-matches["elo_diff"] / 400))

    matches["attack_strength_diff"] = (
        (matches["home_avg_goals_scored"] - matches["away_avg_goals_scored"]) +
        (matches["away_avg_goals_conceded"] - matches["home_avg_goals_conceded"])
    )

    elo_df = pd.DataFrame(elo.items(), columns=["team", "elo"])
    return matches, elo_df