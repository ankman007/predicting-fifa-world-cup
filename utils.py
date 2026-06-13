import pandas as pd

def assign_climate_points(temp):
    # Handle any missing or NaN values safely by grouping them into the moderate tier
    if pd.isna(temp):
        return 1
        
    if temp > 23.0:
        return 2  # Hot climate edge
    elif temp >= 15.0:
        return 1  # Moderate climate
    else:
        return 0  # Cool climate