import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import OneHotEncoder

roty_data = pd.read_csv(r"D:\Personal_Python_Projects\venv\Projects\NBA_ROTY_Prediction\combined_winners_stats.csv")
rookies_data = pd.read_csv(r"D:\Personal_Python_Projects\venv\Projects\NBA_ROTY_Prediction\combined_rookies_stats.csv")

features = ['AGE', 'Draft Pick', 'YEARS', 'G', 'MPG', 'PTS', 'AST', 'RB', 'BLK', 'STL', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'TOV', 'AST/TO', 'PF']

#Drop rows with NaN values (LeBron and Amare Stoudamire didn't go to college)
roty_data.dropna(inplace=True)
rookies_data.dropna(inplace=True)

#Turns POS Column from a string to int
encoder = OneHotEncoder(sparse_output=False)
encoded_positions_roty = encoder.fit_transform(roty_data[['POS']])
encoded_positions_rookies = encoder.transform(rookies_data[['POS']])

encoded_columns = encoder.get_feature_names_out(['POS'])
encoded_positions_roty_df = pd.DataFrame(encoded_positions_roty, columns=encoded_columns)
encoded_positions_rookies_df = pd.DataFrame(encoded_positions_rookies, columns=encoded_columns)

#Adds the POS data back to the original df
roty_data_encoded = pd.concat([roty_data.reset_index(drop=True), encoded_positions_roty_df], axis=1)
rookies_data_encoded = pd.concat([rookies_data.reset_index(drop=True), encoded_positions_rookies_df], axis=1)
features.extend(encoded_columns)

X_past_winners = roty_data_encoded[features].values
average_past_winner = np.mean(X_past_winners, axis=0)

distances = []
for index, rookie_row in rookies_data_encoded.iterrows():
    rookie_features = rookie_row[features].values
    distance = euclidean_distances([average_past_winner], [rookie_features])[0][0]
    distances.append(distance)

#Add distances to rookies_data
rookies_data_encoded['Distance_to_Past_Winner_Avg'] = distances
rookies_data_sorted = rookies_data_encoded.sort_values(by='Distance_to_Past_Winner_Avg')
rookies_data_sorted['Distance_to_Past_Winner_Avg'] = rookies_data_sorted['Distance_to_Past_Winner_Avg'].round(2predictions_csv = r'D:\Personal_Python_Projects\venv\Projects\NBA_ROTY_Prediction\Predicted_ROTY_Winner.csv'

rookies_data_sorted.to_csv(predictions_csv, index=False)
print(f"Predictions saved to: {predictions_csv}")

print("Top Predicted ROTY Candidates:")
print(rookies_data_sorted[['Name', 'Distance_to_Past_Winner_Avg']].head())
