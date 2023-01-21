from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForMaskedLM, Trainer, TrainingArguments

from graph_dataset import GraphDataset

LIST_LOCATION = './data/lists_2nodes.txt'
RESULTS_LOCATION = './data/results_2nodes.txt'


def readInput(lists, tokens):
    lists = open(lists, 'r').readlines()
    tokens = open(tokens, 'r').readlines()
    return lists, tokens

def getEncodings():
    lists, tokens = readInput(LIST_LOCATION, RESULTS_LOCATION)
    train_lists, val_lists, train_tokens, val_tokens = train_test_split(lists, tokens, test_size=.2)

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    train_encode = tokenizer(train_lists, is_split_into_words=True, truncation=True, padding=True)
    val_encode = tokenizer(val_lists, is_split_into_words=True, truncation=True, padding=True)
    # TODO: Need to split the data to have a test set
    # test_encode = tokenizer(test_lists, is_split_into_words=True, truncation=True, padding=True)

    # Convert to dataset objects
    train_dataset = GraphDataset(train_encode, train_tokens)
    val_dataset = GraphDataset(val_encode, val_tokens)
    # test_dataset = GraphDataset(test_encodings, test_labels)
    return train_dataset, val_dataset #,test_dataset

train_data, val_data = getEncodings()

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")

trainer = Trainer(
    model=model,                      # the instantiated 🤗 Transformers model to be trained
    args=training_args,               # training arguments, defined above
    train_dataset=train_data,         # training dataset
    eval_dataset=val_data             # evaluation dataset
)

trainer.train()