import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from transformers import AutoModelForMaskedLM, AutoTokenizer
from datasets import load_dataset


from transformers import pipeline


MODEL_BASE = 'distilbert-base-uncased'
PROMPT = "a normalized synchronous data flow graph with weights (2,3) needs [MASK] tokens for liveness" 

classifier = pipeline("fill-mask", model = MODEL_BASE)

print(classifier(PROMPT))


"""
Prompt:
 a normalized synchronous data flow graph with weights (2,3) needs [MASK] tokens for liveness

Answers:

Distilbert-base-uncased:
1. matching
2. random
3. additional
4. appropriate
5. different

Distilroberta-base
1. 2048
2. additional
3. multiple
4. validation
5. 5
"""