import tensorflow as tf
import tensorflow_gnn as tfgnn

from tensorflow_gnn import runner

# def graph_tensor_from_list(graphs, throughputs):
#     tensors = []
#     for graph, throughput in graphs, throughputs:
#         graph = tfgnn.GraphTensor.from_pieces(
#             context=tfgnn.Context.from_fields(
#                 features={'throughput': throughput}
#             ),
#             node_sets={
#                 "node": tfgnn.NodeSet.from_fields(
#                     sizes=tf.constant([3]),
#                     features={
#                         "weight": tf.constant([[x] for x in graph]),
#                     },
#                 )
#             },
#             edge_sets={
#                 "edge": tfgnn.EdgeSet.from_fields(
#                     sizes=tf.constant([3]),
#                     adjacency=tfgnn.Adjacency.from_indices(
#                         source=("node", tf.constant([-1, 1, 2, 3])),
#                         target=("node", tf.constant([0, 2, 3, 0])),
#                     ),
#                 ),
#             },
#         )
#         tensors.append(graph)
#         writer.write(example.SerializeToString())
#     return tensors

class MyDatasetProvider(runner.DatasetProvider):
    def __init__(self, file_pattern):
        self.file_pattern = file_pattern


    def get_dataset(self, context: tf.distribute.InputContext) -> tf.data.Dataset:
        # Define the feature specifications for the serialized tf.Examples.
        x_list = self.ds.values.tolist()
        y_list = self.ls.values.tolist()
        graph_list = graph_tensor_from_list(x_list, y_list)
        # example_list = [write_example(x) for x in graph_list]
        dataset = tf.convert_to_tensor(graph_list, dtype=None)
        # def generator():
        #     for x in graph_list:
        #         yield x
        # dataset = tf.data.Dataset.from_generator(generator, tf.int32, output_shapes=[None])
        return dataset
