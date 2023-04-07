from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras

from constants import VERBOSE, GRAPH_SIZE

def model_v1():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(GRAPH_SIZE,)),
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
        layers.Dense(8, activation='relu', input_shape=(GRAPH_SIZE,)),
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

def train_tf_dnn(x_train, y_train):
    early_stop = EarlyStopping(monitor='val_loss', patience=10)

    model = model_v2()

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        verbose=VERBOSE,
        epochs=50,
        batch_size=64,
        callbacks=[early_stop]
    )
    return model, history
