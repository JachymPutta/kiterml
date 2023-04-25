from keras.callbacks import EarlyStopping
from kerastuner.tuners import RandomSearch
from sklearn.model_selection import train_test_split
from tensorflow import keras

import constants
from constants import RANDOM_SEED


def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Dense(units=hp.Int('units', min_value=100, max_value=1000, step=100), activation='relu'))
    model.add(keras.layers.Dense(units=hp.Int('units', min_value=500, max_value=5000, step=500), activation='relu'))
    model.add(keras.layers.Dense(units=hp.Int('units', min_value=100, max_value=1000, step=100), activation='relu'))
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

def train_rs_model(model, x_train, y_train):
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        verbose=constants.VERBOSE,
        epochs=50,
        batch_size=64,
        callbacks=[early_stop]
    )
    return model, history
