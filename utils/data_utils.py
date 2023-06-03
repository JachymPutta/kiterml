import math

import pandas as pd
import numpy as np
import tensorflow_gnn as tfgnn
import tensorflow as tf

from sklearn.model_selection import train_test_split

from constants import DATA_LOCATION, MULT_FACTOR, DUP_FACTOR, RANDOM_SEED,\
    GNN_SCHEMA_LOCATION, GNN_TRAIN_LOCATION, GNN_VAL_LOCATION, GNN_TEST_LOCATION, NORMALIZE

# Make sure the data is valid
# def test_data_2_node(data, res):
#     assert (data[0] + data[1] - math.gcd(data[0], data[1])) == res[0]
#     return

def normalize(data):
    # Convert the list of integers to a NumPy array
    data_array = np.array(data)

    # Compute the mean and standard deviation
    mean = np.mean(data_array)
    std = np.std(data_array)

    # Normalize the data
    normalized_data = (data_array - mean) / std

    return normalized_data.tolist()

def preprocess():
    data = []
    results = []

    with open(DATA_LOCATION) as data_file:
        for row in data_file:
            num_list = list(map(int, row.split(',')))
            for m in MULT_FACTOR:
                for i in range(DUP_FACTOR):
                    cur_data = list(map(lambda x: x * m, num_list[:-1]))
                    cur_res = list(map(lambda x: x * m, num_list[-1:]))
                    # test_data_2_node(cur_data, cur_res)
                    data.append(cur_data)
                    results.append(cur_res)

    # TODO: write the train, test data to a file?
    if NORMALIZE:
        data = normalize(data)

    return train_test_split(pd.DataFrame(data), pd.DataFrame(results), test_size=0.15, random_state=RANDOM_SEED)
def preprocess_gnn():
    graph_schema = tfgnn.read_schema(GNN_SCHEMA_LOCATION)
    gtspec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)

    def decode_fn(record_bytes):
        graph = tfgnn.parse_single_example(
            gtspec, record_bytes, validate=True)

        context_features = graph.context.get_features_dict()
        label = context_features.pop('tokens')
        new_graph = graph.replace_features(context=context_features)

        return new_graph, label

    train_ds = tf.data.TFRecordDataset([GNN_TRAIN_LOCATION]).map(decode_fn)
    val_ds = tf.data.TFRecordDataset([GNN_VAL_LOCATION]).map(decode_fn)
    test_ds = tf.data.TFRecordDataset([GNN_TEST_LOCATION]).map(decode_fn)
    return train_ds, val_ds, test_ds
