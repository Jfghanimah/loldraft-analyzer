# Lol Draft Analyzer

The Lol Draft Analyzer is a machine learning model that predicts the outcome of League of Legends (LoL) matches based solely on the champions that are drafted by each team. This model is trained on data from past matches, using tensorflow and incorporating the official Riot Games API to access match data. With an accuracy of 55% over all games and even higher accuracy over subsets of games, this model is a valuable tool for players and teams looking to improve their strategy and performance, as well as for analysts and commentators who want to gain insight into the factors that impact the outcome of a match.

This model is built using a sequential fully connected deep neural network (DNN) with an embedding layer as the input to identify the 162 champions in LoL. The use of an embedding layer allows the DNN model to take into account the relationships between champions and better predict the impact of champion selection on the outcome of a match.

# Running the Model
The model can be run using the example shown in eval_model.py, and you will need to install the following modules, as listed in requirements.txt:

* tensorflow==2.10.1
* numpy==1.23.4
* pandas==1.5.1
* riotwatcher==3.2.3

# Obtaining Match Data
Match data is obtained from the Riot Games API using the riotwatcher module, and a script (data_api.py) has been implemented to populate a save_data folder with all the necessary match data for training the model.

# Key Features
* DNN machine learning model for predicting the outcome of LoL matches based on champion selection
* Trained on data from past matches, incorporating the official Riot Games API to access match data
* 55% accuracy over all games and even higher accuracy over subsets of games
* Built using a sequential fully connected DNN with an embedding layer as the input to help identify and categorize champions
* Embedding layer allows the model to take into account relationships between champions and better predict the impact of champion selection on the outcome of a match.

With the Lol Draft Analyzer, you have access to a powerful tool that can help you gain a competitive edge in the world of League of Legends.
