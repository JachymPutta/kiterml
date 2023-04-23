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