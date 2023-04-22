import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn import runner

from models.gnn.train_ds_provider import MyDatasetProvider


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

    initial_node_states = lambda node_set, node_set_name: node_set["throughput"]
    map_features = tfgnn.keras.layers.MapFeatures(node_sets_fn=initial_node_states)

    task = runner.GraphMeanSquaredError(node_set_name = "nodes")

    trainer = runner.KerasTrainer(
        # strategy = tf.distribute.get_strategy(),
        strategy = tf.distribute.get_strategy(),
    # TODO: hardcoded path badness
        model_dir = "model_output",
        restore_best_weights = False
    )

    run_res = runner.run(
        train_ds_provider = train_ds_provider,
        # train_padding = runner.FitOrSkipPadding(gtspec, train_ds_provider),
        model_fn = model_fn,
        optimizer_fn = tf.keras.optimizers.Adam,
        epochs = 1,
        trainer = trainer,
        task = task,
        gtspec = gtspec,
        global_batch_size = 32,
        feature_processors=[map_features],
        # valid_ds_provider = None
    )

    return run_res.trained_model, None

