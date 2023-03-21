#!/usr/bin/env python
# coding: utf-8


import numpy as np
np.set_printoptions(precision=3, suppress=True)
import os
import tensorflow as tf
import pandas as pd

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow import keras

LIST_LOCATION = './data/lists_3nodes.txt'
RESULTS_LOCATION = './data/results_3nodes.txt'
VERBOSE = True

#Debugging
def log(s):
    if VERBOSE:
        print(s)

# Loading data from the files
def preprocess():
    data = []
    results = []

    with open(LIST_LOCATION) as listFile:
        with open(RESULTS_LOCATION) as resFile:
            for lis, res in zip(listFile, resFile):
                intList = list(map(int, lis[:-1].split(',')))

                results.append(int(res[:-1]))
                data.append(intList)

    return pd.DataFrame(data), pd.DataFrame(results)


# Tiny model
def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))
    return model


# Load the data
data, res = preprocess()
normalizer = tf.keras.layers.Normalization(axis=-1)

eval_results = []
abs_errors = []

data_train, full_test, result_train, full_res = train_test_split(data, res, test_size=0.15)

# Take data in 20% increments
for i in range(15, 100, 20):
    # Get the current segment
    test_sz = i/100
    log("Current test size is: " + str(test_sz))
    _, data_slice, _, result_slice = train_test_split(data_train, result_train, test_size=test_sz)

    # Train/Test split
    log("Data size - training = " + str(len(data_slice)))

    # Build model
    normalizer.adapt(data_slice)
    dnn_model = build_and_compile_model(normalizer)

    # Train
    dnn_model.fit(
        data_slice,
        result_slice,
        validation_split=0.2,
        verbose=0, epochs=100)

    # Eval
    eval_res = dnn_model.evaluate(full_test, full_res, verbose=0)
    log("Eval results for test size " + str(test_sz) + " = " + str(eval_res))
    eval_results.append(eval_res)

    # Get error
    test_predictions = dnn_model.predict(full_test).flatten()
    error = (100 * (test_predictions - full_res.T)) / full_res.T
    abs_error = error.T.sum() / len(full_test)
    log("Error for test size " + str(test_sz) + " = " + str(float(abs_error)))

    abs_errors.append(abs_error)

print("Absolute errors:\n")
print(abs_errors)
print("\n\n")

print("Evaluations:\n")
print(eval_results)
print("\n\n")