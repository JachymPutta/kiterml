import numpy as np
import tensorflow as tf

import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.initializers as initializers
from keras.callbacks import EarlyStopping

import constants


def build_model() -> keras.Model:
    input = layers.Input(shape=(constants.GRAPH_SIZE,))  # input dimension
    hidden1 = layers.Dense(500, activation="relu", kernel_initializer=initializers.he_normal())(input)
    hidden2 = layers.Dense(1000, activation="relu", kernel_initializer=initializers.he_normal())(hidden1)
    hidden3 = layers.Dense(500, activation="relu", kernel_initializer=initializers.he_normal())(hidden2)
    output = layers.Dense(1, kernel_initializer=initializers.Zeros(), activation="linear")(hidden3)

    model = keras.Model(inputs=input, outputs=[output])

    return model

def mse_loss(q_value: tf.Tensor, reward: tf.Tensor) -> tf.Tensor:
    return 0.5 * (q_value - reward) ** 2


# TODO: create a history object during the training
def build_drl_model(x_train, y_train):
    exploration_rate = 0.1
    learning_rate = 0.01
    num_epochs = 3
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    x_train = x_train.values.tolist()
    y_train = y_train.values.tolist()

    model = build_model()

    for i in range(num_epochs):
        for weights, tokens in zip(x_train, y_train):
            with tf.GradientTape() as tape:
                state = tf.constant([weights])
                goal = np.array(tokens)
                q_values = model(state)

                epsilon = np.random.rand()
                if epsilon <= exploration_rate:
                    action = np.random.choice(len(goal))
                else:
                    action = np.argmax(q_values)

                q_value = q_values[0, action]
                loss_value = mse_loss(q_value, goal)
                grads = tape.gradient(loss_value[0], model.trainable_variables)

                opt.apply_gradients(zip(grads, model.trainable_variables))
    return model