#!/usr/bin/env python
# coding: utf-8

from utils.data_utils import preprocess
from models.tf_dnn import train_tf_dnn
from utils.eval_utils import eval_iter, eval_tf_dnn
from utils.plot import plot_all_histories

x_train, x_test, y_train, y_test = preprocess()

histories, evals = eval_iter(train_tf_dnn, eval_tf_dnn, x_train, x_test, y_train, y_test)

plot_all_histories(histories)

