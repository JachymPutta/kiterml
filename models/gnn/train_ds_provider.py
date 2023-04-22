import tensorflow as tf
import tensorflow_gnn as tfgnn

from tensorflow_gnn import runner, write_example
from models.gnn.util import graph_tensor_from_list

class MyDatasetProvider(runner.DatasetProvider):
    def __init__(self, file_pattern, dataset):
        self.file_pattern = file_pattern
        self.ds = dataset
    

    def get_dataset(self, context: tf.distribute.InputContext) -> tf.data.Dataset:
        # Define the feature specifications for the serialized tf.Examples.
        x_list = self.ds.values.tolist()
        graph_list = graph_tensor_from_list(x_list)
        example_list = [write_example(x) for x in graph_list]
        print(type(example_list[0]))
        # dataset = tf.data.Dataset.from_tensors(graph_list)
        def generator():
            for x in graph_list:
                yield x
        dataset = tf.data.Dataset.from_generator(generator, tf.int32, output_shapes=[None])
        return dataset
