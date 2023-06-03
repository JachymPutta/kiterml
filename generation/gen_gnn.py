import tensorflow_gnn as tfgnn
import tensorflow as tf

from sklearn.model_selection import train_test_split

from utils.data_utils import preprocess
from constants import RANDOM_SEED, \
    GNN_TRAIN_LOCATION, GNN_VAL_LOCATION, GNN_TEST_LOCATION, GRAPH_SIZE

def gen_gnn():
    x, x_test, y, y_test = preprocess()
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, random_state=RANDOM_SEED)

    x_train = x_train.values.tolist()
    x_val = x_val.values.tolist()
    x_test = x_test.values.tolist()

    y_train = y_train.values.tolist()
    y_val = y_val.values.tolist()
    y_test = y_test.values.tolist()

    # graph_schema = tfgnn.read_schema(GNN_SCHEMA_LOCATION)
    # gtspec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)

    write_gnns_to_file(x_train, y_train, GRAPH_SIZE, GNN_TRAIN_LOCATION)
    write_gnns_to_file(x_val, y_val, GRAPH_SIZE, GNN_VAL_LOCATION)
    write_gnns_to_file(x_test, y_test, GRAPH_SIZE, GNN_TEST_LOCATION)


def write_gnns_to_file(x_lst, y_lst, graph_size, file):
    if graph_size == 2:
        src = tf.constant([0, 1])
        tgt = tf.constant([1, 0])
    elif graph_size == 4:
        src = tf.constant([0, 1, 2, 3])
        tgt = tf.constant([1, 2, 3, 0])
    elif graph_size == 8:
        src = tf.constant([0, 1, 2, 3, 4, 5, 6, 7])
        tgt = tf.constant([1, 2, 3, 4, 5, 6, 7, 0])
    else:
        raise Exception("write_gnns_to_file: unsupported graph size")
    with tf.io.TFRecordWriter(file) as writer:
        for graph, tokens in zip(x_lst, y_lst):
            weights = tf.constant([[float(x)] for x in graph])
            graph_sum = float(sum(graph))
            init_estimate = tf.constant([[graph_sum] for _ in graph])

            graph = tfgnn.GraphTensor.from_pieces(
                context=tfgnn.Context.from_fields(
                    features={'tokens': [float(tokens[0])]}
                ),
                node_sets={
                    "actor": tfgnn.NodeSet.from_fields(
                        sizes=tf.constant([graph_size]),
                        features={
                            "throughput": weights,
                        },
                    )
                },
                edge_sets={
                    "edge": tfgnn.EdgeSet.from_fields(
                        sizes=tf.constant([graph_size]),
                        adjacency=tfgnn.Adjacency.from_indices(
                            source=("actor", src),
                            target=("actor", tgt),
                        ),
                        features={
                            "throughput": init_estimate,
                        }
                    ),
                },
            )
            example = tfgnn.write_example(graph)
            writer.write(example.SerializeToString())


