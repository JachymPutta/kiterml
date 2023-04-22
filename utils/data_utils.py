import pandas as pd
from sklearn.model_selection import train_test_split

from constants import DATA_LOCATION, MULT_FACTOR, DUP_FACTOR, RANDOM_SEED


def preprocess():
    data = []
    results = []

    with open(DATA_LOCATION) as data_file:
        for row in data_file:
            num_list = list(map(int, row.split(' ')))
            for m in MULT_FACTOR:
                for i in range(DUP_FACTOR):
                    data.append(list(map(lambda x: x * m, num_list[:-1])))
                    results.append(list(map(lambda x: x * m, num_list[-1:])))

    # TODO: write the train, test data to a file?
    return train_test_split(pd.DataFrame(data), pd.DataFrame(results), test_size=0.15, random_state=RANDOM_SEED)

def adj_matrix_cycle(size):
    if size == 4:
        return pd.DataFrame(ADJ_M_4)
    elif size == 2:
        return pd.DataFrame(ADJ_M_2)
    else:
        raise Exception("adj_matrix_cycle: Unsupported graph size")

def list_to_graph(list):
    '''
    The graph neural network needs each graph as a tuple:
    (adjacency_matrix, feature_matrix)
    '''

    graphs = []
    adj_m = adj_matrix_cycle(GRAPH_SIZE)

    for item in list:
        feature_m = np.array(item).reshape(GRAPH_SIZE, 1)
        graphs.append(np.array([adj_m, feature_m]))

    return graphs

    

