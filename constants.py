import os

# General structure
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_LOCATION = os.path.join(ROOT_DIR, 'data', 'data2node.txt')
OUTPUT_FILE = os.path.join(ROOT_DIR, 'results.tmp')
FIG_DIR = os.path.join(ROOT_DIR, 'figs')

# Gnn stuff
GNN_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'gnn')
GNN_OUT_DIR = os.path.join(ROOT_DIR, 'gnn_output')
GNN_SCHEMA_LOCATION = os.path.join(GNN_DATA_DIR, 'gnn_schema.pbtxt')
GNN_TRAIN_LOCATION = os.path.join(GNN_DATA_DIR, 'train.tfrecords')
GNN_VAL_LOCATION = os.path.join(GNN_DATA_DIR, 'val.tfrecords')
GNN_TEST_LOCATION = os.path.join(GNN_DATA_DIR, 'test.tfrecords')

# Model types
TF_MODEL = 'tf'
SKLEARN_MODEL = 'sklearn'

# Logging
VERBOSE = False
TO_FILE = True

# Model Parameters
MULT_FACTOR = [1, 2, 3, 4]
DUP_FACTOR = 1
TRAIN_SET_PERCENTAGE = [15, 35, 55, 75, 95]
GRAPH_SIZE = 2
RANDOM_SEED = 42

#Pretrained model defaults
MODEL_BASE = 'distilbert-base-uncased'
PROMPT = "a normalized synchronous data flow graph with weights (2,3) needs [MASK] tokens for liveness" 
