import pickle

from sklearn.model_selection import train_test_split

from constants import TO_FILE, VERBOSE
from models.gnn import train_gnn
from models.sklearn_dnn import train_sklearn_dnn
from models.sqnn import train_sqnn
from utils.misc import merge_dicts


def eval_model(model, x_test, y_test):
    print("Starging Evaluation")
    # TF only
#     eval_res = model.evaluate(x_test, y_test, verbose=1)
    y_pred = model.predict(x_test).flatten()
    error = (100 * (y_pred - y_test.T)) / y_test.T
    abs_error = abs(error).T.sum() / len(y_test)
        
    return {'abs_error': float(abs_error), 'error_vec': error, 'predictions': y_pred}

def run_eval(model_type, x_train, x_test, y_train, y_test):
    if model_type == 'sklearn':
        x_val_slice, x_slice, y_val_slice, y_slice = train_test_split(x_train, y_train, test_size=0.2)
        model = train_sklearn_dnn(x_slice, y_slice, x_val_slice, y_val_slice)
        eval_res = eval_model(model, x_test, y_test)
        eval_res['train_val_loss'] = (model.loss_curve_, model.validation_scores_)
        eval_res['train_sz'] = (100, len(x_slice))
        if TO_FILE:
            with open('sklearn_model.pkl', 'wb+') as f:
                pickle.dump(model, f)
            with open('sklearn_eval_res.pkl', 'wb+') as f:
                pickle.dump(eval_res, f)
    else:
        if 'sqnn' in model_type:
            model, history = train_sqnn(model_type, x_train, y_train)
            eval_res = eval_model(model, x_test, y_test)
            eval_res['train_val_loss'] = (history.history['loss'], history.history['val_loss'])
            eval_res['train_sz'] = (100, len(x_train))
            if TO_FILE:
                model.save((model_type + '_model'), save_format='tf')
                with open((model_type + '_eval_res.pkl'), 'wb+') as f:
                    pickle.dump(eval_res, f)
        elif model_type == 'gnn':
            train_ds = x_train # Renaming for clarity
            val_ds = x_test # Renaming for clarity
            test_ds = y_train # Renaming for clarity

            model, history = train_gnn(train_ds, val_ds)
            eval_res = eval_model(model, test_ds, y_test)
            eval_res['train_val_loss'] = (history.history['loss'], history.history['val_loss'])
            eval_res['train_sz'] = (100, len(y_test))
            if TO_FILE:
                model.save((model_type + '_model'), save_format='tf')
                with open((model_type + '_eval_res.pkl'), 'wb+') as f:
                    pickle.dump(eval_res, f)

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
