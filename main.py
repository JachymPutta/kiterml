import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from transformers import AutoModelForMaskedLM, AutoTokenizer
from datasets import load_dataset

from graph_dataset import GraphDataset

LIST_LOCATION = './data/lists_2nodes.txt'
RESULTS_LOCATION = './data/results_2nodes.txt'
MODEL_BASE = 'distilbert-base-uncased'


model = AutoModelForMaskedLM.from_pretrained(MODEL_BASE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)

# lists = open(LIST_LOCATION, 'r').readlines()
# tokens = open(RESULTS_LOCATION, 'r').readlines()
list_dataset = load_dataset("text", data_files=LIST_LOCATION)

def tokenize_function(examples):
    result = tokenizer(examples)
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

tokens = tokenize_function(lists)
# print(tokens)


# Use batched=True to activate fast multithreading!
# tokenized_datasets = imdb_dataset.map(
    # tokenize_function, batched=True, remove_columns=["text", "label"]
# )
# tokenized_datasets
# def readInput(lists, tokens):
    # return lists, tokens

# def getEncodings():
    # lists, tokens = readInput(LIST_LOCATION, RESULTS_LOCATION)
    # train_lists, val_lists, train_tokens, val_tokens = train_test_split(lists, tokens, test_size=.2)