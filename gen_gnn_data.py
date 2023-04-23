import tensorflow_gnn as tfgnn
import tensorflow as tf

from sklearn.model_selection import train_test_split

from utils.data_utils import preprocess
from constants import RANDOM_SEED

x, x_test, y, y_test = preprocess()

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, random_state=RANDOM_SEED)

x_train = x_train.values.tolist()
x_val = x_val.values.tolist()
x_test = x_test.values.tolist()

y_train = y_train.values.tolist()
y_val = y_val.values.tolist()
y_test = y_test.values.tolist()

graph_schema = tfgnn.read_schema("data/gnn/gnn_schema.pbtxt")
gtspec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)


def write_gnns_to_file(x_lst, y_lst, file):
    with tf.io.TFRecordWriter(file) as writer:
        for graph, throughput in zip(x_lst, y_lst):
            graph = tfgnn.GraphTensor.from_pieces(
                context=tfgnn.Context.from_fields(
                    features={'throughput': throughput}
                ),
                node_sets={
                    "node": tfgnn.NodeSet.from_fields(
                        sizes=tf.constant([4]),
                        features={
                            "weight": tf.constant([[x] for x in graph]),
                        },
                    )
                },
                edge_sets={
                    "edge": tfgnn.EdgeSet.from_fields(
                        sizes=tf.constant([4]),
                        adjacency=tfgnn.Adjacency.from_indices(
                            source=("node", tf.constant([0, 1, 2, 3])),
                            target=("node", tf.constant([1, 2, 3, 0])),
                        ),
                    ),
                },
            )
            example = tfgnn.write_example(graph)
            writer.write(example.SerializeToString())


write_gnns_to_file(x_train, y_train, 'data/gnn/train.tfrecords')
write_gnns_to_file(x_val, y_val, 'data/gnn/val.tfrecords')
write_gnns_to_file(x_test, y_test, 'data/gnn/test.tfrecords')
