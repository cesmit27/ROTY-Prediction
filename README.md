# ROTY-Prediction
Uses college stats of past ROTY Winners in the NBA to predict who will win the award for the 2024/25 season using euclidean distance.


prediction.py and Predicted_ROTY_Winner.csv are the main pieces here, everything else is for holding the stats and names of the players for use by prediction.py. The jupyter notebook files were used to clean up CSV files, web scrape names, draft pick number, etc of these players.

Data is sourced from https://www.sports-reference.com, https://www.basketball-reference.com, and Wikipedia


# UPDATE AFTER THE SEASON
So the season is done (several months ago at this point, whoops) so I wanted to see how this model did. 

TL;DR it did ok.

The winner of the ROTY award ended up being Stephon Castle of the Spurs, congratualtions to him. The model had him ranked 17th, with the distance to past winners being 41.06. Model performance was jsut ok here.
The runner up, Zaccharie Risacher of the Hawks, was ranked 4th with a distance to past winners of 30.01. The model did a pretty good job here!
Third place was Jaylen Wells of the Grizzlies, who was ranked 32nd by the model with a distance to past winners of 59.48. The model did not do a great job here.
Fourth place was Alex Sarr of the Wizards, who was ranked 19th by the model with a distance to past winners of 42.33. The model did alright here.
Fifth place was Zach Edey of the Grizzlies. Nice job having two rookies in contention, Memphis! Zach was ranked 50th by the model with a distance to past winners of 90.42. My best guess for why this was so off was due to Zach being a multi year starter for Purdue, whereas a lot of ROTY winners seem to be 1 and done type playes.
Sixth place was Kel'el Ware of the Heat. He was ranked 3rd by the model with a distance to past winners of 27.82. Pretty good perfomance by the model here!
Seventh place was Matas Buzelis of the Bulls. He was ranked 43rd by the model with a distance to past winner of 75.88. This is not as bad as the Zach Edey whiff but still not a good job by the model.
Lastly, receiving one third place vote for ROTY was Jared McCain of the 76ers. This was a case that the model didn't really account for. Jared was looking like a strong contender for the award but suffered a season ending meniscus injury in December, so he only played for about a third of the season. The model had him ranked 11th, with a distance to past winners of 37.44. So him getting 8th in voting is actually really close to what the model predicted, but had he stayed healthy and kept playing as well as he was (big "ifs"), he likely would have finished higher in award voting and the model would have been less accurate.

Overall, a decent perfomance by the model. The best prediction it had was definitley in its prediction for Zaccharie Risacher. Congratulations to all 8 of these players for being in contention for this award.
