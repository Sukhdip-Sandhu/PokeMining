import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def read_csv(pokemon_data, combat_data, test_data):
    pokemon_df = pd.read_csv(pokemon_data)
    combat_df = pd.read_csv(combat_data)
    test_df = pd.read_csv(test_data)
    return [pokemon_df, combat_df, test_df]


def clean_up_dataset(pokemon):
    # clean up pokemon data (rename columns and add in missing pokemon name)
    pokemon = pokemon.rename(index=str, columns={'#': 'ID', 'Sp. Atk': 'Sp_Atk', 'Sp. Def': 'Sp_Def'})
    pokemon['Name'] = pokemon['Name'].fillna('Primeape')
    return pokemon


def id_to_name_dictionary(pokemon):
    return dict(zip(pokemon['ID'], pokemon['Name']))


def id_to_stats_dictionary(pokemon):
    stats = pokemon.drop(['Name', 'Type 1', 'Type 2', 'Generation','Legendary'], axis=1)
    return stats.set_index('ID').T.to_dict('list')


def get_features(pokestats, battle_date, MODE):
    # initialize x matrix to contain difference in stats
    x = np.zeros([len(battle_date),len(pokestats[1])])

    # calculate the difference in stats for every battling pokemon
    for i, battle in battle_date.iterrows():
        x[i] = np.array(pokestats[battle['First_pokemon']]) - np.array(pokestats[battle['Second_pokemon']])

    # if training data, calculate y labels
    if MODE is 'Train':
        battle_date['Winner'] = battle_date['First_pokemon'] != battle_date['Winner']
        y = battle_date.Winner.astype(int)
        return x, y
    else:
        return x


def train_neural_network(x_training_data, y_training_data):
    # split data into 75% training and 25% testing
    x_train, x_cv, y_train, y_cv = train_test_split(x_training_data, y_training_data, test_size=0.25, random_state=42)
    # train a neural net classifier
    classifier = MLPClassifier(alpha=1)
    model = classifier.fit(x_train, y_train)

    # calculate accuracy: (~94%)
    accuracy_prediction = model.predict(x_cv)
    print('Neural Network Accuracy: ', accuracy_score(accuracy_prediction , y_cv))
    return model


def battle_outcomes(all_combats, predictions, pokedex):
    # convert pokemon IDs of results back to pokemon names
    all_combats['Winner'] = predictions
    all_combats['Winner'][all_combats['Winner'] == 1] = all_combats['Second_pokemon']
    all_combats['Winner'][all_combats['Winner'] == 0] = all_combats['First_pokemon']
    battle_outcome = all_combats[['First_pokemon', 'Second_pokemon', 'Winner']].replace(pokedex)
    return battle_outcome


def calculate_win_rates(results):
    # calculate the win percenage of every pokemon in sorted order
    win_counts = results['Winner'].value_counts()
    win_percentage = win_counts / 1600
    win_percentage = win_percentage.sort_values()
    return win_percentage


def main(pokemon_data, combat_data, test_data):
    # read csv data
    [pokemon, combats, all_combats_data] = read_csv(pokemon_data, combat_data, test_data)

    # clean up dataset
    pokemon = clean_up_dataset(pokemon)

    # create dictionaries
    pokedex = id_to_name_dictionary(pokemon)
    pokestats = id_to_stats_dictionary(pokemon)

    # get training features and labels
    [x_train, y_labels] = get_features(pokestats, combats, MODE='Train')

    # training neural net mlp classifier
    neural_network_model = train_neural_network(x_train, y_labels)

    # extract features from testing dataset
    x_test = get_features(pokestats, all_combats_data, MODE='Test')

    # get predictions based off neural net model
    predictions = neural_network_model.predict(x_test)

    # determine the outcomes of the all the battles
    battle_results = battle_outcomes(all_combats_data, predictions, pokedex)

    # determine the win rates for each pokemon
    win_rates = calculate_win_rates(battle_results)

    # save data to csv to further analysis
    battle_results.to_csv(r'./Results/all_vs_all_battle_results.csv', index=False)
    win_rates.to_csv(r'./Results/win_rates.csv')


if __name__ == "__main__":
    # paths to csv files
    pokemon_data_csv_path = r'./Data/pokemon.csv'
    pokemon_combat_csv_path = r'./Data/combats.csv'
    pokemon_prediction_csv_path = r'./Data/all_vs_all.csv'

    # pass file paths to main
    main(pokemon_data_csv_path, pokemon_combat_csv_path, pokemon_prediction_csv_path)
