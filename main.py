#!/usr/bin/env python
# coding: utf-8
import pickle
import tensorflow as tf

from constants import TF_MODEL, SKLEARN_MODEL, OUTPUT_FILE

from utils.data_utils import preprocess
from utils.plot import plot_loss_curves, plot_loss_curve
from utils.misc import write_results, write_config
from utils.eval_utils import run_eval, run_eval_iter, eval_model
from models.random_search import run_random_search

x_train, x_test, y_train, y_test = preprocess()

# Random search
# rs_model = run_random_search(x_train, y_train)
# evals_rs = run_eval(TF_MODEL,  x_train, x_test, y_train, y_test)

# Training models
evals_tf = run_eval(TF_MODEL,  x_train, x_test, y_train, y_test)
# evals_sklearn = run_eval(SKLEARN_MODEL, x_train, x_test, y_train, y_test)

# Record results -- only for iterative training
# write_results(TF_MODEL + "_" + OUTPUT_FILE, evals_tf)
# write_results(SKLEARN_MODEL + "_" + OUTPUT_FILE, evals_sklearn)

# Plotting
# plot_loss_curve(TF_MODEL, evals_tf['train_val_loss'])
# plot_loss_curve(SKLEARN_MODEL, evals_sklearn['train_val_loss'])

# Load saved models
# with open('sklearn_model.pkl', 'rb') as f:
#     sklearn_model = pickle.load(f)
#     eval_model(sklearn_model, x_test, y_test)
# tf_model = tf.keras.models.load_model("tf_model.h5")
# eval_model(tf_model, x_test, y_test)

# Load evals - from file
#   with open('tf_eval_res.pkl', 'rb') as f:
#       evals_tf = pickle.load(f)
#       plot_loss_curve(TF_MODEL, evals_tf['train_val_loss'])
#       write_results(TF_MODEL + "_" + OUTPUT_FILE, evals_tf)

#   with open('sklearn_eval_res.pkl', 'rb') as f:
#       evals_tf = pickle.load(f)
#       plot_loss_curve(SKLEARN_MODEL, evals_sklearn['train_val_loss'])
#       write_results(SKLEARN_MODEL + "_" + OUTPUT_FILE, evals_sklearn)
