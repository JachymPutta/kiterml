#!/usr/bin/env python
# coding: utf-8

from constants import TF_MODEL, SKLEARN_MODEL, OUTPUT_FILE

from utils.data_utils import preprocess
from utils.plot import plot_loss_curves
from utils.misc import write_results
from utils.eval_utils import run_eval

x_train, x_test, y_train, y_test = preprocess()

# Training all the models
evals_tf = run_eval(TF_MODEL,  x_train, x_test, y_train, y_test)
evals_sklearn = run_eval(SKLEARN_MODEL, x_train, x_test, y_train, y_test)

# Record results
write_results(TF_MODEL + "_" + OUTPUT_FILE, evals_tf)
write_results(SKLEARN_MODEL + "_" + OUTPUT_FILE, evals_sklearn)

# Plotting
plot_loss_curves(TF_MODEL, evals_tf['train_val_loss'])
plot_loss_curves(SKLEARN_MODEL, evals_sklearn['train_val_loss'])
