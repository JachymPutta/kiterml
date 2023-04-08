from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from constants import TRAIN_SET_PERCENTAGE
from utils.misc import merge_dicts

from models.sklearn_dnn import train_sklearn_dnn
from models.tf_dnn import train_tf_dnn

def better_eval_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_prob)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'auc_roc': auc_roc}

def eval_model(model, x_test, y_test):
    print("Starging Evaluation")
    # TF only
#     eval_res = model.evaluate(x_test, y_test, verbose=1)
    y_pred = model.predict(x_test).flatten()
    error = (100 * (y_pred - y_test.T)) / y_test.T
    abs_error = abs(error).T.sum() / len(x_test)
        
    return {'abs_error': float(abs_error), 'error_vec': error, 'predictions': y_pred}

def run_eval(model_type, x_train, x_test, y_train, y_test):
    evals = []

    match model_type:
        case 'tf':
            for i in TRAIN_SET_PERCENTAGE:
                test_sz = i/100
                _, x_slice, _, y_slice = train_test_split(x_train, y_train, test_size=test_sz)
                
                model, history = train_tf_dnn(x_slice, y_slice)
                eval_res = eval_model(model, x_test, y_test)
                eval_res['train_val_loss'] = (history.history['loss'], history.history['val_loss'])
                eval_res['train_sz'] = (test_sz, len(x_slice))
                evals.append(eval_res)
        case 'sklearn':
            for i in TRAIN_SET_PERCENTAGE:
                test_sz = i/100
                x_val_slice, x_slice, y_val_slice, y_slice = train_test_split(x_train, y_train, test_size=test_sz)
                model, history = train_sklearn_dnn(x_slice, y_slice, x_val_slice, y_val_slice)
                eval_res = eval_model(model, x_test, y_test)
                eval_res['train_val_loss'] = (model.loss_curve_, model.validation_scores_)
                eval_res['train_sz'] = (test_sz, len(x_slice))
                evals.append(eval_res)
    return merge_dicts(evals)
