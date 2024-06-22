import torch
import numpy as np


from tqdm import *

if torch.cuda.is_available():
	device = torch.device("cuda")
else:
	device = torch.device("cpu")
print(device)

def build_knns(k, batch_size, embed_path, knn_path):
    dataset = np.load(embed_path)
    data_shape = dataset.shape
    k = min(k, data_shape[0])
    dataset_size = data_shape[0]

    for start_id in tqdm(range(0,dataset_size, batch_size)):
        cur_batch_size = min(batch_size, dataset_size - start_id)
        dist = torch.zeros((cur_batch_size, dataset_size), dtype=torch.float32).to(device)
        ind = torch.zeros((cur_batch_size, dataset_size,), dtype=torch.int).to(device)
        q_batch_ids = np.arange(cur_batch_size) + start_id
        q_batch = torch.tensor(dataset[q_batch_ids], dtype=torch.float32).to(device)
        # print(q_batch.shape)
        for k_start_id in range(0,dataset_size, batch_size):
            k_batch_ids = np.arange(min(batch_size, dataset_size - k_start_id)) + k_start_id
            k_batch = torch.tensor(dataset[k_batch_ids], dtype=torch.float32).to(device)
            dist[:,k_batch_ids] = torch.matmul(q_batch, k_batch.T)
        
        d, id = torch.topk(dist, k)
        np.save(knn_path + f'batch{start_id//batch_size}.npy', id.cpu())

import sys
import argparse

if __name__ == '__main__':
    cmd = argparse.ArgumentParser('Preprocessing-embed_building arguments')
    cmd.add_argument('--k', type=int, default=1024)
    cmd.add_argument('--batch-size', type=int, default=1000)
    cmd.add_argument('--embed-path', type=str)
    cmd.add_argument('--knn-path', type=str)
    args = cmd.parse_args(sys.argv[1:])

    build_knns(args.k, args.batch_size, args.embed_path, args.knn_path)
