import tensorflow as tf
import tensorflow_gnn as tfgnn


def _build_model(graph_tensor_spec, node_dim=16, edge_dim=16, message_dim=64, next_state_dim=64,
                 num_classes=2, num_message_passing=3, l2_regularization=5e-4, dropout_rate=0.5):
    input_graph = tf.keras.layers.Input(type_spec=graph_tensor_spec)
    graph = input_graph.merge_batch_to_components()

    def set_initial_node_state(node_set, *, node_set_name):
        # Since we only have one node set, we can ignore node_set_name.
        return tf.keras.layers.Dense(node_dim)(node_set['actor'])

    def set_initial_edge_state(edge_set, *, edge_set_name):
        return tf.keras.layers.Dense(edge_dim)(edge_set['edge'])

    graph = tfgnn.keras.layers.MapFeatures(node_sets_fn=set_initial_node_state, edge_sets_fn=set_initial_edge_state)(
        graph)

    def dense(units, activation="relu"):
        """A Dense layer with regularization (L2 and Dropout)."""
        regularizer = tf.keras.regularizers.l2(l2_regularization)
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
                "actor": tfgnn.keras.layers.NodeSetUpdate(
                    {"throughput": tfgnn.keras.layers.SimpleConv(
                        sender_edge_feature=tfgnn.HIDDEN_STATE,
                        message_fn=dense(message_dim),
                        reduce_type="sum",
                        receiver_tag=tfgnn.TARGET)},
                    tfgnn.keras.layers.NextStateFromConcat(dense(next_state_dim)))}
        )(graph)

    readout_features = tfgnn.keras.layers.Pool(
        tfgnn.CONTEXT, "mean", node_set_name="actor")(graph)
    logits = tf.keras.layers.Dense(1)(readout_features)

    return tf.keras.Model(inputs=[input_graph], outputs=[logits])


def train_gnn(train_ds, val_ds):
    # Sanity check for the dataset
    # g, y = train_ds.take(1).get_single_element()
    # print(g.node_sets['actor'].features['throughput'])
    # print(y)

    batch_size = 32
    train_ds_batched = train_ds.batch(batch_size=batch_size).repeat()
    val_ds_batched = val_ds.batch(batch_size=batch_size)

    model_input_graph_spec, label_spec = train_ds.element_spec
    del label_spec  # Unused.
    model = _build_model(model_input_graph_spec)

    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.BinaryAccuracy(threshold=0.),
               tf.keras.metrics.BinaryCrossentropy(from_logits=True)]

    model.compile(tf.keras.optimizers.Adam(), loss=loss, metrics=metrics)

    print(model.summary())

    history = model.fit(train_ds_batched,
                        steps_per_epoch=10,
                        epochs=200,
                        validation_data=val_ds_batched)
    return model, history

    # task = runner.GraphMeanSquaredError(node_set_name="nodes")
    #
    # trainer = runner.KerasTrainer(
    #     # strategy = tf.distribute.get_strategy(),
    #     strategy=tf.distribute.get_strategy(),
    #     model_dir = GNN_OUT_DIR
    # )
    #
    # run_res = runner.run(
    #     train_ds_provider=train_ds_provider,
    #     # train_padding = runner.FitOrSkipPadding(gtspec, train_ds_provider),
    #     model_fn = model_fn,
    #     optimizer_fn=tf.keras.optimizers.Adam,
    #     epochs=1,
    #     trainer=trainer,
    #     task=task,
    #     gtspec=gtspec,
    #     global_batch_size=32,
    #     feature_processors=[map_features],
    #     valid_ds_provider=val_ds_provider
    # )
    #
    # return run_res.trained_model, None
