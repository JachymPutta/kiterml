#!/usr/bin/env python
# coding: utf-8


import numpy as np
np.set_printoptions(precision=3, suppress=True)
import os
import tensorflow as tf
import pandas as pd
import sys

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow import keras

# LIST_LOCATION = './data/lists_3nodes.txt'
# RESULTS_LOCATION = './data/results_3nodes.txt'
# LIST_LOCATION = './data/lists4node.txt'
# RESULTS_LOCATION = './data/results4node.txt'
VERBOSE = True
TO_FILE = True
OUTPUT_FILE = 'results.out'

if len(sys.argv) > 1:
    DATA_LOCATION = sys.argv[1]
else:
    DATA_LOCATION = 'data2node.txt'

#Debugging
def log(s):
    if VERBOSE:
        print(s)
    if TO_FILE:
        if os.path.exists(OUTPUT_FILE):
            append_write = 'a'
        else:
            append_write = 'w'
        fh = open(OUTPUT_FILE, append_write)
        fh.write(s + '\n')
        fh.close()

# Loading data from the files
def preprocess():
    data = []
    results = []

    with open(DATA_LOCATION) as data_file:
        for row in data_file:
            num_list = list(map(int, row.split(' ')))
            data.append(num_list[:-1])
            results.append(num_list[-1:])
    return pd.DataFrame(data), pd.DataFrame(results)


# Tiny model
def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(1000, input_dim=4,  activation='relu'),
        layers.Dense(500, activation='relu'),
        layers.Dense(1, activation='linear')
    ])

    model.compile(loss='mean_squared_error',
                  optimizer='adam')
    return model


# Load the data
print("Loading data...")
data, res = preprocess()
print("Data loaded!")
normalizer = tf.keras.layers.Normalization(axis=-1)

eval_results = []
abs_errors = []

print("Train/Test split")
data_train, full_test, result_train, full_res = train_test_split(data, res, test_size=0.15)

# Take data in 20% increments
for i in range(15, 100, 20):
    print("Starting iteration")

    # Get the current segment
    test_sz = i/100
    _, data_slice, _, result_slice = train_test_split(data_train, result_train, test_size=test_sz)

    log("Training set is " + str(test_sz) + " of the total data i.e. " + str(len(data_slice)) + " points.")

    # Build model
    normalizer.adapt(data_slice)
    dnn_model = build_and_compile_model(normalizer)

    # Train
    print("Starting training")
    dnn_model.fit(
        data_slice,
        result_slice,
        validation_split=0.2,
        verbose=1, epochs=50)
    print("Training done!")

    # Eval
    print("Starging Evaluation")
    eval_res = dnn_model.evaluate(full_test, full_res, verbose=1)
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
