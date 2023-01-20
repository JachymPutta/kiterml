from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast

from graph_dataset import GraphDataset

LIST_LOCATION = './data/lists_2nodes.txt'
RESULTS_LOCATION = './data/results_2nodes.txt'


def readInput(lists, tokens):
    lists = open(lists, 'r').readlines()
    tokens = open(tokens, 'r').readlines()
    return lists, tokens

lists, tokens = readInput(LIST_LOCATION, RESULTS_LOCATION)
train_lists, val_lists, train_tokens, val_tokens = train_test_split(lists, tokens, test_size=.2)

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encode = tokenizer(train_lists, truncation=True, padding=True)
val_encodings = tokenizer(val_lists, truncation=True, padding=True)

# Need to split the data to have a test set
# test_encodings = tokenizer(test_texts, truncation=True, padding=True)
# test_dataset = IMDbDataset(test_encodings, test_labels)

train_dataset = IMDbDataset(train_encodings, train_tokens)
val_dataset = IMDbDataset(val_encodings, val_tokens)
