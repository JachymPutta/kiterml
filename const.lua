KITER_PATH = '../kiter/Release/bin/kiter'
TWO_NODE = './data/2/%d_%d.xml'
THREE_NODE = './data/3/%d_%d_%d.xml'
FOUR_NODE = './data/4/%d_%d_%d_%d.xml'
ACTOR_TYPE = 'A'

ONE = arg[2]
TWO = arg[3]
THREE = arg[4]
FOUR = arg[5]

GRAPH_TYPE = FOUR_NODE
LIM_TOKEN = 30
LENGTH = 5
RANGE = 5
INIT_TOKEN = arg[1] == nil and 500 or arg[1]
-- local SEED = arg[2] == nil and 42 or arg[2]
