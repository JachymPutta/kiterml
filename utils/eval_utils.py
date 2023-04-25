import pickle

from sklearn.model_selection import train_test_split
import os

import constants

from models.gnn import train_gnn
from models.drl import build_drl_model
from models.random_search import run_random_search, train_rs_model
from models.sklearn_dnn import train_sklearn_dnn
from models.sqnn import train_sqnn
from utils.misc import merge_dicts


def eval_model(model, x_test, y_test):
    # TF only
#     eval_res = model.evaluate(x_test, y_test, verbose=1)
    y_pred = model.predict(x_test).flatten()
    error = (100 * (y_pred - y_test.T)) / y_test.T
    abs_error = abs(error).T.sum() / len(y_test)
        
    return {'abs_error': float(abs_error), 'error_vec': error, 'predictions': y_pred}

def write_eval_res(model_type, model, history, train_size, x_test, y_test):
    eval_res = eval_model(model, x_test, y_test)
    eval_res['train_val_loss'] = (history.history['loss'], history.history['val_loss'])
    eval_res['train_sz'] = (100, train_size)
    if constants.TO_FILE:
        model_name = model_type + '_model'
        res_name = model_type + '_eval_res.pkl'
        model.save(os.path.join(constants.ROOT_DIR, model_name ), save_format='tf')
        with open(os.path.join(constants.ROOT_DIR, res_name), 'wb+') as f:
            pickle.dump(eval_res, f)
    return eval_res


def run_eval(model_type, x_train, x_test, y_train, y_test):
    if model_type == 'sklearn':
        x_val_slice, x_slice, y_val_slice, y_slice = train_test_split(x_train, y_train, test_size=0.2)

        model = train_sklearn_dnn(x_slice, y_slice, x_val_slice, y_val_slice)

        # Eval
        eval_res = eval_model(model, x_test, y_test)
        eval_res['train_val_loss'] = (model.loss_curve_, model.validation_scores_)
        eval_res['train_sz'] = (100, len(x_slice))

        # Write out results
        if constants.TO_FILE:
            model_name = model_type + '_model.pkl'
            res_name = model_type + '_eval_res.pkl'
            with open(os.path.join(constants.ROOT_DIR, model_name), 'wb+') as f:
                pickle.dump(model, f)
            with open(os.path.join(constants.ROOT_DIR, res_name), 'wb+') as f:
                pickle.dump(eval_res, f)

    elif 'sqnn' in model_type:
        model, history = train_sqnn(model_type, x_train, y_train)
        eval_res = write_eval_res(model_type, model, history, len(x_train), x_test, y_test)

    elif model_type == 'gnn':
        # Renaming for clarity
        train_ds = x_train
        val_ds = x_test
        test_ds = y_train

        model, history = train_gnn(train_ds, val_ds)
        eval_res = write_eval_res(model_type, model, history, len(y_test), test_ds, y_test)

    elif model_type == 'random_search':
        model = run_random_search(x_train, y_train)
        trained_model, history = train_rs_model(model, x_train, y_train)
        eval_res = write_eval_res(model_type, trained_model, history, len(y_test), x_test, y_test)
    elif model_type == 'drl':
        model = build_drl_model(x_train, y_train)
        model.compile(optimizer='adam', loss='mse', metrics=['mse'])
        eval_res = eval_model(model, x_test, y_test)
        # eval_res['train_val_loss'] = (history.history['loss'], history.history['val_loss'])
        eval_res['train_sz'] = (100, len(y_test))
        model_name = model_type + '_model'
        if constants.TO_FILE:
            model.save(os.path.join(constants.ROOT_DIR, model_name ), save_format='tf')
            # with open((model_type + '_eval_res.pkl'), 'wb+') as f:
            #     pickle.dump(eval_res, f)
    else:
        raise Exception("run_eval: Unknown graph type")

    return merge_dicts([eval_res])

# def run_eval_iter(model_type, x_train, x_test, y_train, y_test):
#     evals = []
#
#     if model_type == 'tf':
#             for i in train_set_percentage:
#                 test_sz = i/100
#                 _, x_slice, _, y_slice = train_test_split(x_train, y_train, test_size=test_sz)
#                 
#                 model, history = train_sqnn(3, x_slice, y_slice)
#                 eval_res = eval_model(model, x_test, y_test)
#                 eval_res['train_val_loss'] = (history.history['loss'], history.history['val_loss'])
#                 eval_res['train_sz'] = (test_sz, len(x_slice))
#                 evals.append(eval_res)
#     elif model_type == 'sklearn':
#             for i in train_set_percentage:
#                 test_sz = i/100
#                 x_val_slice, x_slice, y_val_slice, y_slice = train_test_split(x_train, y_train, test_size=test_sz)
#                 model, history = train_sklearn_dnn(x_slice, y_slice, x_val_slice, y_val_slice)
#                 eval_res = eval_model(model, x_test, y_test)
#                 eval_res['train_val_loss'] = (model.loss_curve_, model.validation_scores_)
#                 eval_res['train_sz'] = (test_sz, len(x_slice))
#                 evals.append(eval_res)
#         
#     return merge_dicts(evals)
