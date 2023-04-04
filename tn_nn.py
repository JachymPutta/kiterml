#!/usr/bin/env python
# coding: utf-8


import os
import sys
import tensorflow as tf
import pandas as pd
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import matplotlib.pyplot as plt

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
FIG_DIR = 'figs/'
MULT_FACTOR = [1,10,100,1000]
DUP_FACTOR = 1
TRAIN_SET_PERCENTAGE = [15, 35, 55, 75, 95]

if len(sys.argv) > 1:
    DATA_LOCATION = sys.argv[1]
else:
    DATA_LOCATION = 'data/data2node.txt'

###############################################################################
#           UTILITIES
###############################################################################
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

def plot_all_predictions(preds, act_vals):
    fig, axs = plt.subplots(len(TRAIN_SET_PERCENTAGE), sharex=True, sharey=True)
    
    fig.suptitle('Predictions / Actual Values')
    fig.supxlabel('True Values')
    fig.supylabel('Predictions')

    for i in range(len(TRAIN_SET_PERCENTAGE)):
        axs[i].scatter(act_vals.T, preds[i])
        axs[i].set_title(str(TRAIN_SET_PERCENTAGE[i]) + "% of data")
    
    if TO_FILE:
        fig.savefig(FIG_DIR + 'all_predictions.png')
    
def plot_all_errors(errors):
    fig, axs = plt.subplots(len(TRAIN_SET_PERCENTAGE), sharex=True, sharey=True)
    
    fig.suptitle('Percentage errors')
    fig.supxlabel('True Values')
    fig.supylabel('Predictions')

    for i in range(len(TRAIN_SET_PERCENTAGE)):
        axs[i].hist(errors[i].T, bins=25)
        axs[i].set_title(str(TRAIN_SET_PERCENTAGE[i]) + "% of data")
        
    if TO_FILE:
        fig.savefig(FIG_DIR + 'all_errors.png')
        
        
def plot_all_histories(histories):
    fig, axs = plt.subplots(len(TRAIN_SET_PERCENTAGE))
    
    fig.suptitle('Histories')
    fig.supxlabel('Accuracy')
    fig.supylabel('Loss')
    
    for i in range(len(TRAIN_SET_PERCENTAGE)):
        axs[i].plot(histories[i].history['loss'])
        axs[i].plot(histories[i].history['val_loss'])
        axs[i].legend(['train', 'val'], loc='upper left')

    if TO_FILE:
        fig.savefig(FIG_DIR + 'all_histories.png')

###############################################################################

def preprocess():
    data = []
    results = []

    with open(DATA_LOCATION) as data_file:
        for row in data_file:
            num_list = list(map(int, row.split(' ')))
            for m in MULT_FACTOR:
                for i in range(DUP_FACTOR):
                    data.append(list(map(lambda x: x * m, num_list[:-1])))
                    results.append(list(map(lambda x: x * m, num_list[-1:])))
    return pd.DataFrame(data), pd.DataFrame(results)



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



def train_iter(data_train, full_test, result_train, full_res):

    eval_results = []
    abs_percentage_errors = []
    train_sizes = []
    predictions = []
    full_errors = []
    histories = []
    
    for i in TRAIN_SET_PERCENTAGE:
        print("Starting iteration")

        # Get the current segment
        test_sz = i/100
        _, data_slice, _, result_slice = \
        train_test_split(data_train, result_train, test_size=test_sz)
        
        print("Training set is " + str(test_sz) + " of the total data i.e. " \
            + str(len(data_slice)) + " points.")
        train_sizes.append((test_sz, len(data_slice)))

        # Build model
        normalizer.adapt(data_slice)
        dnn_model = build_and_compile_model(normalizer)

        # Train
        print("Starting training")
        history = dnn_model.fit(
            data_slice,
            result_slice,
            validation_split=0.2,
            verbose=VERBOSE,
            epochs=20)
        histories.append(history)
        print("Training done!")

        # Eval
        print("Starging Evaluation")
        eval_res = dnn_model.evaluate(full_test, full_res, verbose=1)
        print("Eval results for test size " + str(test_sz) + " = " + str(eval_res))
        eval_results.append(float(eval_res))

        # Get error
        test_predictions = dnn_model.predict(full_test).flatten()
        predictions.append(test_predictions)
        
        error = (100 * (test_predictions - full_res.T)) / full_res.T
        full_errors.append(error)
        
        abs_error = abs(error).T.sum() / len(full_test)
        print("Error for test size " + str(test_sz) + " = " + str(float(abs_error)))

        abs_percentage_errors.append(float(abs_error))
        print("\n\n")
        
    return abs_percentage_errors, full_errors, predictions, eval_results, train_sizes, histories

# Load the data
print("Loading data...")
d,r = preprocess()
print("Data loaded!")
normalizer = tf.keras.layers.Normalization(axis=-1)

data_train, full_test, result_train, full_res = train_test_split(d, r, test_size=0.15)

all_percentage_errors, all_full_errors, all_predictions, all_evals, all_train_sizes, all_histories = \
train_iter(data_train, full_test, result_train, full_res )


# Write out results
log("Evaluation results")
log('=' * 80)
log("Data metadata:\n" +
    "  Location: " + DATA_LOCATION +
    "\n  Size: "+ str(len(d)) + " points\n" +
    "  MULT_FACTOR = " + str(MULT_FACTOR) + "\n")
log("Percentage of data used for training:")

s = ""
for sz in all_train_sizes:
    s += str(sz[0]) + "%% (" + str(sz[1]) + ")  " 

log(s + "\n")
log("Average Percentage Errors:" )
log(' '.join(map(str, all_percentage_errors)))
log("")
log("Evaluation results:")
log(' '.join(map(str, all_evals)))
log('=' * 80)
log("\n\n")

# Plots
plot_all_predictions(all_predictions, full_res)
plot_all_errors(all_full_errors)
plot_all_histories(all_histories)
