import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn import runner

from constants import GNN_OUT_DIR, GNN_TRAIN_LOCATION, GNN_VAL_LOCATION, GNN_SCHEMA_LOCATION

def _build_model(
    graph_tensor_spec,
    node_dim=16,
    edge_dim=16,
    message_dim=64,
    next_state_dim=64,
    num_classes=2,
    num_message_passing=3,
    l2_regularization=5e-4,
    dropout_rate=0.5,
):
  input_graph = tf.keras.layers.Input(type_spec=graph_tensor_spec)

  graph = input_graph.merge_batch_to_components()

  initial_node_state = lambda node_set, node_set_name: node_set["throughput"]
  initial_edge_state = lambda edge_set, edge_set_name: edge_set["edges"]

  graph = tfgnn.keras.layers.MapFeatures(
      node_sets_fn=initial_node_state, edge_sets_fn=initial_edge_state)(
          graph)

  def dense(units, activation="relu"):
    regumodel_fnlarizer = tf.keras.regularizers.l2(l2_regularization)
    return tf.keras.Sequential([
        tf.keras.layers.Dense(
            units,
            activation=activation,
            kernel_regularizer=regularizer,
            bias_regularizer=regularizer),
        tf.keras.layers.Dropout(dropout_rate)
    ])

  for i in range(num_message_passing):
    graph = tfgnn.keras.layers.GraphUpdate(
        node_sets={
            "atoms": tfgnn.keras.layers.NodeSetUpdate(
                {"bonds": tfgnn.keras.layers.SimpleConv(
                     sender_edge_feature=tfgnn.HIDDEN_STATE,
                     message_fn=dense(message_dim),
                     reduce_type="sum",
                     receiver_tag=tfgnn.TARGET)},
                tfgnn.keras.layers.NextStateFromConcat(dense(next_state_dim)))}
    )(graph)

  readout_features = tfgnn.keras.layers.Pool(
      tfgnn.CONTEXT, "mean", node_set_name="atoms")(graph)

  logits = tf.keras.layers.Dense(1)(readout_features)

  return tf.keras.Model(inputs=[input_graph], outputs=[logits])

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
      convolver.Convolve()
  ])

def train_gnn():
    graph_schema = tfgnn.read_schema(GNN_SCHEMA_LOCATION)
    gtspec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)

    train_ds_provider = runner.TFRecordDatasetProvider(filenames=[GNN_TRAIN_LOCATION])
    val_ds_provider = runner.TFRecordDatasetProvider(filenames=[GNN_VAL_LOCATION])

    initial_node_states = lambda node_set, node_set_name: node_set["throughput"]
    map_features = tfgnn.keras.layers.MapFeatures(node_sets_fn=initial_node_states)

    task = runner.GraphMeanSquaredError(node_set_name="nodes")

    trainer = runner.KerasTrainer(
        # strategy = tf.distribute.get_strategy(),
        strategy=tf.distribute.get_strategy(),
        model_dir = GNN_OUT_DIR
    )

    run_res = runner.run(
        train_ds_provider=train_ds_provider,
        # train_padding = runner.FitOrSkipPadding(gtspec, train_ds_provider),
        model_fn = model_fn,
        optimizer_fn=tf.keras.optimizers.Adam,
        epochs=1,
        trainer=trainer,
        task=task,
        gtspec=gtspec,
        global_batch_size=32,
        feature_processors=[map_features],
        valid_ds_provider=val_ds_provider
    )

    return run_res.trained_model, None
