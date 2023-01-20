from sklearn.model_selection import train_test_split

LIST_LOCATION = './data/lists_2nodes.txt'
RESULTS_LOCATION = './data/results_2nodes.txt'


def readInput(lists, tokens):
    lists = open(lists, 'r').readlines()
    tokens = open(tokens, 'r').readlines()
    return lists, tokens

lists, tokens = readInput(LIST_LOCATION, RESULTS_LOCATION)
train_lists, test_lists, train_tokens, test_tokens = train_test_split(lists, tokens, test_size=.2)