from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras

from constants import VERBOSE

def build_and_compile_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(2,)),
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

    model = build_and_compile_model()

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        verbose=VERBOSE,
        epochs=20,
        batch_size=32,
        callbacks=[early_stop]
    )
    return model, history
