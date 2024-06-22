
import numpy as np
import faiss
import pickle as pkl

path = '/public/home/tengzhh2022/GPST/preprocess/in-context-pretraining/output/embed/valid/tokenizer-facebook_contriever_msmarco/seq_len-5100/chunk_len-510/model-facebook_contriever_msmarco/train.jsonl.npy'


data_shape = (222546, 768)
dataset = np.memmap(path, dtype=np.float32, shape=data_shape)
nlist = 32678
nprobe = 64
d = 768
m = 256
assert(d % m == 0)
quantizer = faiss.IndexFlatL2(d)

# index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
index = faiss.index_factory(768, 'IVF32768,PQ256')
# 8 specifies that each sub-vector is encoded as 8 bits

train_size = 1277952
train_index = np.random.randint(0,data_shape[0], (train_size,))
train_batch = dataset[train_index]

index.train(train_batch)
index.add(dataset)
 
f_Index=open('IndexIVFPQ.pkl','wb')
pkl.dump(index, f_Index)

# k = 11
# D, I = index.search(dataArray[:5], k)     # search

