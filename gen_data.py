import tensorflow_gnn as tfgnn
import tensorflow as tf

from utils.data_utils import preprocess

x_train, x_test, y_train, y_test = preprocess()

x_list = x_train.values.tolist()
y_list = y_train.values.tolist()

graph_schema = tfgnn.read_schema("models/gnn/gnn_schema.pbtxt")
gtspec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)

with tf.io.TFRecordWriter('tfgnn_data/train.tfrecords') as writer:
    for graph, throughput in zip(x_list, y_list):
        print(throughput)
        graph = tfgnn.GraphTensor.from_pieces(
            context=tfgnn.Context.from_fields(
                features={'throughput': throughput}
            ),
            node_sets={
                "node": tfgnn.NodeSet.from_fields(
                    sizes=tf.constant([3]),
                    features={
                        "weight": tf.constant([[x] for x in graph]),
                    },
                )
            },
            edge_sets={
                "edge": tfgnn.EdgeSet.from_fields(
                    sizes=tf.constant([3]),
                    adjacency=tfgnn.Adjacency.from_indices(
                        source=("node", tf.constant([-1, 1, 2, 3])),
                        target=("node", tf.constant([0, 2, 3, 0])),
                    ),
                ),
            },
        )
        example = tfgnn.write_example(graph)
        writer.write(example.SerializeToString())
