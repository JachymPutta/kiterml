# Model types
TF_MODEL = 'tf'
SKLEARN_MODEL = 'sklearn'

# Logging
VERBOSE = False
TO_FILE = True

# Directories
OUTPUT_FILE = 'result.tmp'
FIG_DIR = 'figs/'
DATA_LOCATION = 'data/data4node.txt'

# Model Parameters
MULT_FACTOR = [1,2,3,4]
DUP_FACTOR = 1
TRAIN_SET_PERCENTAGE = [15, 35, 55, 75, 95]
GRAPH_SIZE = 4
RANDOM_SEED = 42

#Pretrained model defaults
MODEL_BASE = 'distilbert-base-uncased'
PROMPT = "a normalized synchronous data flow graph with weights (2,3) needs [MASK] tokens for liveness" 
