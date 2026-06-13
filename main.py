import pandas as pd

# 1. Load your full datasets
df_groups = pd.read_csv('data/world_cup_groups.csv')
df_results = pd.read_csv('data/international_football_results.csv')

# --- NEW: ADD MANUAL OVERRIDES FOR HISTORICAL NAMES ---
# This ensures 'Czech Republic', 'Turkey', and 'Cape Verde' in your results file 
# match the lowercase formats 'czechia', 'türkiye', and 'cabo verde' from your groups file.
manual_overrides = {
    'czech republic': 'czechia',
    'turkey': 'türkiye',
    'cape verde': 'cabo verde'
}

# Create a base lowercase mapping dictionary from the groups file
name_mapping = dict(zip(df_groups['country'].str.strip().str.lower(), df_groups['country']))

# Merge your manual overrides into the mapping dictionary
# This tells the script: "If you see 'czech republic', overwrite it with 'czechia'"
name_mapping.update(manual_overrides)


# 2. Get all unique teams from the results file (both home and away columns)
# Apply the manual overrides temporarily during the check so they report as "Found"
results_teams = set(df_results['home_team'].str.strip().str.lower().dropna()) \
          .union(set(df_results['away_team'].str.strip().str.lower().dropna()))

# We simulate the replacement for the check report
results_teams_mapped = {name_mapping.get(team, team) for team in results_teams}

print("--- TEAM PRESENCE CHECK REPORT ---")
missing_teams = []
present_teams = []

# Check each team from the groups file against the mapped results teams
for country in df_groups['country'].dropna().unique():
    cleaned_country = str(country).strip().lower()
    if cleaned_country in results_teams_mapped:
        present_teams.append(country)
    else:
        missing_teams.append(country)

print(f"Total teams in World Cup Groups: {len(df_groups['country'].unique())}")
print(f"Teams found in Results File: {len(present_teams)}")
print(f"Teams NOT found in Results File: {len(missing_teams)}")

if missing_teams:
    print("\n⚠️ These teams from your groups file were NOT found in the results file:")
    print(missing_teams)
else:
    print("\n✅ All teams from the groups file exist perfectly in the results file!")


# 3. HARMONIZATION STEP
def harmonize_name(team_name):
    if pd.isna(team_name):
        return team_name
    cleaned = str(team_name).strip().lower()
    # Overwrite if found in your groups mapping or manual overrides; keep original if not found
    return name_mapping.get(cleaned, team_name)

# Overwrite both columns in the results dataset
df_results['home_team'] = df_results['home_team'].apply(harmonize_name)
df_results['away_team'] = df_results['away_team'].apply(harmonize_name)

# Save the newly corrected file
df_results.to_csv('results_harmonized.csv', index=False)
print("\n🎉 Harmonization complete! Saved to 'results_harmonized.csv'.")