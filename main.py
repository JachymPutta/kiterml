#!/usr/bin/env python
# coding: utf-8

from utils.data_utils import preprocess
from utils.eval_utils import eval_iter, eval_model
from utils.plot import plot_all_histories

from models.tf_dnn import train_tf_dnn
from models.sklearn_dnn import train_sklearn_dnn

x_train, x_test, y_train, y_test = preprocess()

# Training all the models
# hist_tf, evals_tf = eval_iter(train_tf_dnn, eval_model, x_train, x_test, y_train, y_test)
# loss_curves_sklearn, evals_sklearn = eval_iter(train_sklearn_dnn, eval_model, x_train, x_test, y_train, y_test)

# Plotting
# plot_histories(hist_tf)
# plot_loss_curves(loss_curves_sklearn)


