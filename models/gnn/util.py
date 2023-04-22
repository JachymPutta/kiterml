import tensorflow as tf
import tensorflow_gnn as tfgnn


def graph_tensor_from_list(list_of_graphs):
    graphs = []
    for item in list_of_graphs:
        graph = tfgnn.GraphTensor.from_pieces(
            node_sets={
                "node": tfgnn.NodeSet.from_fields(
                    sizes=tf.constant([4]),
                    features={
                        "weight": tf.constant([[x] for x in item]),
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
        graphs.append(graph)
    return graphs

