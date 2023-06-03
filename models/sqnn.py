from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

import constants

def model_v0():
    model = keras.Sequential([
        layers.Dense(1000, activation='relu', input_shape=(constants.GRAPH_SIZE,)),
        layers.Dense(500, activation='relu'),
        layers.Dense(1, activation='linear')
    ])

    model.compile(loss='mean_squared_error',
                  optimizer='adam')
    return model

def model_v1():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(constants.GRAPH_SIZE,)),
        layers.BatchNormalization(),
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(1, activation='linear')
    ])

    model.compile(loss='mean_squared_error',
                  optimizer='adam')
    return model

def model_v2():
    model = keras.Sequential([
        layers.Dense(8, activation='relu', input_shape=(constants.GRAPH_SIZE,)),
        layers.Dense(16, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(1, activation='linear')
    ])

    model.compile(loss='mean_squared_error',
                  optimizer='adam')
    return model

def model_v3():
    model = keras.Sequential([
        layers.Dense(500, activation='relu', input_shape=(constants.GRAPH_SIZE,)),
        layers.Dense(1000, activation='relu'),
        layers.Dense(10000, activation='relu'),
        layers.Dense(1000, activation='relu'),
        layers.Dense(500, activation='relu'),
        layers.Dense(1, activation='linear')
    ])

    model.compile(loss='mean_squared_error',
                  optimizer='adam')
    return model

def train_sqnn(version, x_train, y_train):
    early_stop = EarlyStopping(monitor='val_loss', patience=10)

    if version == 'sqnn0':
        model = model_v0()
    elif version == 'sqnn1':
        model = model_v1()
    elif version == 'sqnn2':
        model = model_v2()
    elif version == 'sqnn3':
        model = model_v3()
    else:
        raise Exception(f"train_sqnn: Unsupported model type ${version}")

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        verbose=constants.VERBOSE,
        epochs=150,
        batch_size=64,
        callbacks=[early_stop]
    )
    return model, history


