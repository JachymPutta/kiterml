import tensorflow as tf

from tensorflow_gnn import runner

class MyDatasetProvider(runner.DatasetProvider):
    def __init__(self, file_pattern, dataset):
        self.file_pattern = file_pattern
        self.ds = dataset
    

    def get_dataset(self, context: tf.distribute.InputContext) -> tf.data.Dataset:
      # Define the feature specifications for the serialized tf.Examples.
      x_list = x_train.values.tolist()
      graphs = graph_tensor_from_list(self.ds)
      dataset = tf.data.Dataset.from_generator(lambda: t, tf.int32, output_shapes=[None])
      return dataset
