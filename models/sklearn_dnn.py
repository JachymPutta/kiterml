from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

def build_and_compile_model():
    model = MLPClassifier(
        solver='adam',
        hidden_layer_sizes=(128,128),
        early_stopping=False,
        max_iter=300
    )
    return model 

def train_sklearn_dnn(x_train, y_train):
    model = build_and_compile_model()
    model.fit(x_train, y_train.iloc[:,0])
    return model, model.loss_curve_

