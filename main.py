#!/usr/bin/env python
# coding: utf-8
import argparse
# Disabling Tensorflow warnings -- no GPU etc.
import os
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from constants import OUTPUT_FILE

from utils.data_utils import preprocess, preprocess_gnn
from utils.plot import plot_loss_curve
from utils.misc import write_results
from utils.eval_utils import run_eval, eval_model  # run_eval_iter
from models.random_search import run_random_search
from models.sqnn import train_rs_model

parser = argparse.ArgumentParser(description="KiterML: training DNNs for cyclic SDF graph liveness estimations")

model_group = parser.add_mutually_exclusive_group(required=True)
model_group.add_argument('--train_model', help='Train model from scratch',
                         choices=['sqnn0', 'sqnn1', 'sqnn2', 'sqnn3', 'sklearn', 'sqnn_random_search', 'gnn'])
model_group.add_argument('--load_pkl', help='Load a pretrained model from a pickle file')
model_group.add_argument('--load_tf', help='Load a pretrained tensorflow model from folder')

data_group = parser.add_argument_group()
data_group.add_argument('--data', help='Location of the data')
data_group.add_argument('--graph_size', help='Size of the graph')

parser.add_argument('--plot', help="Create plots", action='store_true')
parser.add_argument('--verbose', help="More verbose output", action='store_true')
parser.add_argument('--to_file', help="Store trained model in a file", action='store_true')
parser.add_argument('--write_results', help="Write results to file", action='store_true')

args = parser.parse_args()

# Set globals
TO_FILE = args.to_file
VERBOSE = args.verbose

if args.data is not None:
    DATA_LOCATION = args.data
if args.graph_size is not None:
    GRAPH_SIZE = int(args.graph_size)

x_train, x_test, y_train, y_test = preprocess()

if args.load_tf is not None:
    tf_model = tf.keras.models.load_model(args.load_tf)
    evals = eval_model(tf_model, x_test, y_test)
elif args.load_pkl is not None:
    with open(args.load_pkl, 'rb') as f:
        model = pickle.load(f)
        evals = eval_model(model, x_test, y_test)
else:
    if args.train_model == 'random_search':
        model = run_random_search(x_train, y_train)
        trained_model, _ = train_rs_model(model, x_train, y_train)
        evals = eval_model(trained_model, x_test, y_test)
    elif args.train_model == 'gnn':
        train_ds, val_ds, test_ds = preprocess_gnn()
        evals = run_eval(args.train_model, train_ds, val_ds, test_ds, y_test)

    else:
        evals = run_eval(args.train_model, x_train, x_test, y_train, y_test)

if args.write_results:
    write_results(args.train_model + "_" + OUTPUT_FILE, evals)

if args.plot:
    plot_loss_curve(args.train_model, evals['train_val_loss'])
