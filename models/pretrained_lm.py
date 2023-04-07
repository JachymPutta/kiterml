from transformers import pipeline

def run_pretrained(model_base = MODEL_BASE, prompt = PROMPT):
    classifier = pipeline("fill-mask", model = MODEL_BASE)
    return classifier(PROMPT)

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
