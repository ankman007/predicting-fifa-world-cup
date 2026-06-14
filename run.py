import pandas as pd

# Load your file (replace 'your_file.csv' with the actual filename or path)
df = pd.read_csv('data/final_training_data.csv')

# Get all unique tournaments
unique_tournaments = df['tournament'].unique()

# Display the unique values
print(unique_tournaments)