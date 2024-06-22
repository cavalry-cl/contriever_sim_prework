import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import AutoTokenizer, AutoModel

model_path = 'model'
model_name = 'facebook/contriever-msmarco'

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=os.path.join(model_path, "tokenizer"))

model = AutoModel.from_pretrained(
    model_name,
    cache_dir=os.path.join(model_path, "model"))