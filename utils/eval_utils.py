from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from constants import TRAIN_SET_PERCENTAGE

def better_eval_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_prob)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'auc_roc': auc_roc}

def eval_sklearn_dnn(model, x_test, y_test):
    y_pred = model.predict(x_test).flatten()
    error = (100 * (y_pred - y_test.T)) / y_test.T
    abs_error = abs(error).T.sum() / len(x_test)
    return {'abs_error': abs_error, 'error_vec': error, 'predictions': y_pred}

def eval_tf_dnn(model, x_test, y_test):
    print("Starging Evaluation")
    eval_res = model.evaluate(x_test, y_test, verbose=1)
    # print("Eval results for test size " + str(test_sz) + " = " + str(eval_res))
    y_pred = model.predict(x_test).flatten()
    error = (100 * (y_pred - y_test.T)) / y_test.T
    abs_error = abs(error).T.sum() / len(x_test)
    # print("Error for test size " + str(test_sz) + " = " + str(float(abs_error)))
        
    return {'abs_error': abs_error, 'error_vec': error, 'predictions': y_pred, 'eval_res': eval_res}

def eval_model(model, x_test, y_test):
    y_pred = model.predict(x_test).flatten()
    error = (100 * (y_pred - y_test.T)) / y_test.T
    abs_error = abs(error).T.sum() / len(x_test)

    # print("Eval results for test size " + str(test_sz) + " = " + str(eval_res))
    # print("Error for test size " + str(test_sz) + " = " + str(float(abs_error)))
        
    return {'abs_error': abs_error, 'error_vec': error, 'predictions': y_pred}

def eval_iter(train_fun, eval_fun, x_train, x_test, y_train, y_test):
    histories = []
    evals = []

    for i in TRAIN_SET_PERCENTAGE:
        test_sz = i/100
        _, x_slice, _, y_slice = train_test_split(x_train, y_train, test_size=test_sz)
        model, history = train_fun(x_slice, y_slice)
        histories.append(history)
        eval_res = eval_fun(model, x_test, y_test)
        evals.append(eval_res)
    return histories, evals



    
