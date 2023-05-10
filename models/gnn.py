import tensorflow as tf
import tensorflow_gnn as tfgnn


def _build_model(graph_tensor_spec, node_dim=4,message_dim=500, next_state_dim=500,
                num_message_passing=3, l2_regularization=5e-4, dropout_rate=0.2):

    input_graph = tf.keras.layers.Input(type_spec=graph_tensor_spec)
    graph = input_graph.merge_batch_to_components()

    def set_initial_node_state(node_set, *, node_set_name):
        # Since we only have one node set, we can ignore node_set_name.
        return tf.keras.layers.Dense(node_dim)(node_set['throughput'])

    graph = tfgnn.keras.layers.MapFeatures(node_sets_fn=set_initial_node_state)(graph)

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
        conv_layer = tfgnn.keras.layers.SimpleConv(
            sender_edge_feature='throughput',
            message_fn=dense(message_dim),
            reduce_type="sum",
            receiver_tag=tfgnn.TARGET)

        next_st = tfgnn.keras.layers.NextStateFromConcat(dense(next_state_dim))

        graph = tfgnn.keras.layers.GraphUpdate (
            node_sets = {
                'actor': tfgnn.keras.layers.NodeSetUpdate (
                    { 'edge': conv_layer },
                    next_st
                )}
        )(graph)

    readout_features = tfgnn.keras.layers.Pool(
        tfgnn.CONTEXT, "mean", node_set_name="actor")(graph)
    logits = tf.keras.layers.Dense(1, activation='linear')(readout_features)

    return tf.keras.Model(inputs=[input_graph], outputs=[logits])


def train_gnn(train_ds, val_ds):
    # Sanity check for the dataset
    # g, y = train_ds.take(1).get_single_element()
    # print(g.node_sets['actor'].features['throughput'])
    # print(g.edge_sets['edge'].features['throughput'])

    batch_size = 32
    train_ds_batched = train_ds.batch(batch_size=batch_size).repeat()
    val_ds_batched = val_ds.batch(batch_size=batch_size)

    model_input_graph_spec, label_spec = train_ds.element_spec
    del label_spec # Unused.
    model = _build_model(model_input_graph_spec)

    model.compile(tf.keras.optimizers.Adam(), loss='mean_squared_error')

    history = model.fit(train_ds_batched,
                        steps_per_epoch=10,
                        epochs=100,
                        validation_data=val_ds_batched)
    return model, history
