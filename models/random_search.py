from kerastuner.tuners import RandomSearch
from sklearn.model_selection import train_test_split
from tensorflow import keras

from constants import RANDOM_SEED


def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu'))
    model.add(keras.layers.Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu'))
    model.add(keras.layers.Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu'))
    model.add(keras.layers.Dense(1, activation='linear'))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='mse', metrics=['mae'])
    return model

def run_random_search(x, y):
    x_train, x_val, y_train,  y_val = train_test_split(x, y, random_state=RANDOM_SEED)

    tuner = RandomSearch(build_model,
                         objective='val_loss',
                         max_trials=3,
                         executions_per_trial=3,
                         project_name='random_search')

    tuner.search(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return best_model

