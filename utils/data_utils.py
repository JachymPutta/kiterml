import pandas as pd
import tensorflow_gnn as tfgnn
import tensorflow as tf

from sklearn.model_selection import train_test_split

from constants import DATA_LOCATION, MULT_FACTOR, DUP_FACTOR, RANDOM_SEED,\
    GNN_SCHEMA_LOCATION, GNN_TRAIN_LOCATION, GNN_VAL_LOCATION, GNN_TEST_LOCATION


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
