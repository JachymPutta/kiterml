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

VERBOSE = False
TO_FILE = True
OUTPUT_FILE = 'result.tmp'
MULT_FACTOR = 100

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
            for i in range(MULT_FACTOR):
                data.append(num_list[:-1])
                results.append(num_list[-1:])
    return pd.DataFrame(data), pd.DataFrame(results)


# Tiny model
def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(100, input_dim=4,  activation='relu'),
        layers.Dense(50, activation='relu'),
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
abs_percentage_errors = []
train_sizes = []

print("Train/Test split")
data_train, full_test, result_train, full_res = train_test_split(data, res, test_size=0.15)

# Take data in 20% increments
for i in range(15, 100, 20):
    print("Starting iteration")

    # Get the current segment
    test_sz = i/100
    _, data_slice, _, result_slice = train_test_split(data_train, result_train, test_size=test_sz)

    print("Training set is " + str(test_sz) + " of the total data i.e. " + str(len(data_slice)) + " points.")
    train_sizes.append((test_sz, len(data_slice)))

    # Build model
    normalizer.adapt(data_slice)
    dnn_model = build_and_compile_model(normalizer)

    # Train
    print("Starting training")
    dnn_model.fit(
        data_slice,
        result_slice,
        validation_split=0.2,
        verbose=VERBOSE,
        epochs=20)
    print("Training done!")

    # Eval
    print("Starging Evaluation")
    eval_res = dnn_model.evaluate(full_test, full_res, verbose=1)
    print("Eval results for test size " + str(test_sz) + " = " + str(eval_res))
    eval_results.append(float(eval_res))

    # Get error
    test_predictions = dnn_model.predict(full_test).flatten()
    error = (100 * (test_predictions - full_res.T)) / full_res.T
    abs_error = abs(error).T.sum() / len(full_test)
    print("Error for test size " + str(test_sz) + " = " + str(float(abs_error)))

    abs_percentage_errors.append(float(abs_error))


log("Evaluation results")
log('=' * 80)
log("Data metadata:\n" +
    "  Location: " + DATA_LOCATION +
    "\n  Size: "+ str(len(data)) + " points\n" +
    "  MULT_FACTOR = " + str(MULT_FACTOR) + "\n")
log("Percentage of data used for training:")

s = ""
for sz in train_sizes:
    s += str(sz[0]) + "%% (" + str(sz[1]) + ")  " 

log(s + "\n")
log("Average Percentage Errors:" )
log(' '.join(map(str, abs_percentage_errors)))
log("")
log("Evaluation results:")
log(' '.join(map(str, eval_results)))
log('=' * 80)
log("\n\n")
