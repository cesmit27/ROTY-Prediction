import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

roty_data = pd.read_csv(r"D:\Personal_Python_Projects\venv\Projects\NBA_ROTY_Prediction\combined_winners_stats.csv")
rookies_data = pd.read_csv(r"D:\Personal_Python_Projects\venv\Projects\NBA_ROTY_Prediction\combined_rookies_stats.csv")

features = ['AGE', 'Draft Pick', 'YEARS', 'G', 'GS', 'MPG', 'PTS', 'AST', 'RB', 'BLK', 'STL', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'TOV', 'AST/TO', 'PF', 'SOS', 'NAT_CH']

#Function to fill NaNs with median and mark original NaNs
#(LeBron and Amare Stoudamire didn't go to college, and some stats like SOS weren't avaialble for international players)
def fill_nan_with_median_and_mark(df, feature_list):
    df_filled = df.copy()
    for feature in feature_list:
        mean_value = df[feature].mean().round(2)
        df_filled[feature] = df[feature].fillna(mean_value)
        df_filled[f'{feature}_was_nan'] = df[feature].isnull().astype(int)
    return df_filled

# Fill NaNs with median and mark original NaNs
roty_data_filled = fill_nan_with_median_and_mark(roty_data, features)
rookies_data_filled = fill_nan_with_median_and_mark(rookies_data, features)

# Extract the feature values
X_past_winners = roty_data_filled[features].values
average_past_winner = np.mean(X_past_winners, axis=0)

distances = []
for index, rookie_row in rookies_data_filled.iterrows():
    rookie_features = rookie_row[features].values
    distance = euclidean_distances([average_past_winner], [rookie_features])[0][0]
    distances.append(distance)

rookies_data_filled['Distance_to_Past_Winner'] = distances
rookies_data_sorted = rookies_data_filled.sort_values(by='Distance_to_Past_Winner')
rookies_data_sorted['Distance_to_Past_Winner'] = rookies_data_sorted['Distance_to_Past_Winner'].round(2)

cols = list(rookies_data_sorted.columns)
cols = ['Name', 'Distance_to_Past_Winner'] + [col for col in cols if col not in ['Name', 'Distance_to_Past_Winner']]
rookies_data_sorted = rookies_data_sorted[cols]

predictions_csv = r'D:\Personal_Python_Projects\venv\Projects\NBA_ROTY_Prediction\Predicted_ROTY_Winner.csv'
rookies_data_sorted.to_csv(predictions_csv, index=False)
print(f"Predictions saved to: {predictions_csv}")

print("Top Predicted ROTY Candidates:")
print(rookies_data_sorted[['Draft Pick', 'Name', 'Distance_to_Past_Winner']].head())