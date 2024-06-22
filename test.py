import numpy as np
import json
from fairseq.data.indexed_dataset import MMapIndexedDataset

seen = set()

with open('result/start209985/gpt2-train_sort.json','r') as f:
    x = json.load(f)
    for i in x:
        for j in i['doc_seq']:
            seen.add(j)
print(len(seen))
