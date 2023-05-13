import os

# Model Parameters
MULT_FACTOR = [1]
DUP_FACTOR = 1
TRAIN_SET_PERCENTAGE = [15, 35, 55, 75, 95]
RANDOM_SEED = 42
GRAPH_SIZE = 2
DATA_FILE = 'data' + str(GRAPH_SIZE) + 'node.txt'
OUT_DATA_FILE = 'data' + str(GRAPH_SIZE) + 'node-min_max.txt'

# Logging
VERBOSE = False
TO_FILE = False
NORMALIZE = True

# General structure
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_LOCATION = os.path.join(ROOT_DIR, 'data', DATA_FILE)
OUTPUT_FILE = os.path.join(ROOT_DIR, 'results.tmp')
FIG_DIR = os.path.join(ROOT_DIR, 'figs')

# Gnn stuff
GNN_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'gnn' , str(GRAPH_SIZE))
GNN_SCHEMA_LOCATION = os.path.join(ROOT_DIR, 'data', 'gnn', 'gnn_schema.pbtxt')
GNN_OUT_DIR = os.path.join(ROOT_DIR, 'gnn_output')
GNN_TRAIN_LOCATION = os.path.join(GNN_DATA_DIR, 'train.tfrecords')
GNN_VAL_LOCATION = os.path.join(GNN_DATA_DIR, 'val.tfrecords')
GNN_TEST_LOCATION = os.path.join(GNN_DATA_DIR, 'test.tfrecords')

#Pretrained model defaults
MODEL_BASE = 'distilbert-base-uncased'
PROMPT = "a normalized synchronous data flow graph with weights (2,3) needs [MASK] tokens for liveness" 
