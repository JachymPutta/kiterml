import tensorflow_gnn as tfgnn
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow_gnn import runner

from constants import GRAPH_SIZE
from utils.data_utils import adj_matrix_cycle
from models.gnn.train_ds_provider import MyDatasetProvider


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


def model_fn(gtspec: tfgnn.GraphTensorSpec):
  """Builds a simple GNN with `ConvGNNBuilder`."""
  convolver = tfgnn.keras.ConvGNNBuilder(
      lambda edge_set_name, receiver_tag: tfgnn.keras.layers.SimpleConv(
          lambda: tf.keras.layers.Dense(32, activation="relu"),
          "sum",
          receiver_tag=receiver_tag,
          sender_edge_feature=tfgnn.HIDDEN_STATE),
      lambda node_set_name: tfgnn.keras.layers.NextStateFromConcat(
          lambda: tf.keras.layers.Dense(32, activation="relu")),
      receiver_tag=tfgnn.SOURCE)
  return tf.keras.Sequential([
      convolver.Convolve() for _ in range(4)  # Message pass 4 times.
  ])

def train_gnn(x_train, y_train):

    # TODO: hardcoded path badness
    graph_schema = tfgnn.read_schema("models/gnn/gnn_schema.pbtxt")
    gtspec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)

    train_ds_provider = MyDatasetProvider("...", x_train)

    trainer = None # TODO: implement
    task = None  # TODO: implement
    map_features = None # TODO: implement
    valid_ds_provider = None # TODO: implement

    runner.run(
        train_ds_provider = train_ds_provider,
        train_padding = runner.FitOrSkipPadding(gtspec, train_ds_provider),
        model_fn = model_fn,
        optimizer_fn = tf.keras.optimizers.Adam,
        epochs = 4,
        trainer = trainer,
        task = task,
        gtspec = gtspec,
        global_batch_size = 64,
        feature_processors=[map_features],
        valid_ds_provider = valid_ds_provider
    )
    exit()

    return model, history

