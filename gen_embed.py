import torch
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import AutoTokenizer, AutoModel
import argparse
import sys
from fairseq.data.indexed_dataset import MMapIndexedDataset
from fairseq.data.indexed_dataset import MMapIndexedDatasetBuilder
import numpy as np
from tqdm import *

PAD = 0
if torch.cuda.is_available():
	device = torch.device("cuda")
else:
	device = torch.device("cpu")
print(device) 
    
def embed_batch(inputs, model):
    # print(inputs.shape)
    inputs = inputs.int().to(device)
    attention_mask = (inputs != PAD)
    attention_mask = attention_mask.to(device)
    # print(inputs.device, attention_mask.device)
    outputs = model(input_ids=inputs, attention_mask=attention_mask)
    
    # Mean pooling
    def mean_pooling(token_embed, mask):
        token_embed = token_embed.masked_fill(~mask[..., None].bool(), 0.)
        seg_embed = token_embed.sum(dim=1) / mask.sum(dim=1)[..., None]
        return seg_embed

    embeddings = mean_pooling(outputs[0], attention_mask)
    del inputs
    del attention_mask
    del outputs
    return embeddings

def generate_embeddings(data_path, embed_dir, model_dir, batch_size):
    embed_dim = 768
    model = AutoModel.from_pretrained(model_dir)
    model.eval()
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        print(n_gpu)
        model = torch.nn.DataParallel(model,range(n_gpu))
        model = model.to(device)
        # print('#',model.device)
    dataset = MMapIndexedDataset(data_path)
    dataset_size = len(dataset.sizes)
    print(f'dataset_size={dataset_size}')
    dim = dataset[0].shape[0]
    cur_seg = torch.empty([0, dim])
    offset = 0


    seg_embed_path = embed_dir + '_embed.npy'
    
    embedded_batch = np.zeros(shape=(dataset_size, embed_dim), dtype=np.float32)

    last_check_point = 0
    offset = 0

    for (seg_id, seg) in tqdm(enumerate(dataset)):
        seg = seg.clone()
        for idx in range(dim):
            if seg[idx] < 0:
                seg[idx] = -seg[idx]
        cur_seg = torch.cat((cur_seg, seg.reshape(1,-1)), dim=0)
        if cur_seg.shape[0] >= batch_size:
            embed_result = embed_batch(cur_seg, model).cpu().detach().numpy()
            embedded_batch[offset:offset + cur_seg.shape[0], :] = embed_result
            offset += cur_seg.shape[0]
            cur_seg = torch.empty([0, dim])
        if seg_id % 10000 == 9999:
            np.save(embed_dir+f'_batch{last_check_point}~{seg_id}_embed.npy',embedded_batch[last_check_point:seg_id+1,:])
            last_check_point = seg_id + 1
    if cur_seg.shape[0] > 0:
        embed_result = embed_batch(cur_seg, model).cpu().detach().numpy()
        embedded_batch[offset:] = embed_result
    print(embed_dim, dim)
    np.save(seg_embed_path, embedded_batch)
    print('Embedding done!')

if __name__ == '__main__':
    cmd = argparse.ArgumentParser('Preprocessing-embed_building arguments')
    cmd.add_argument('--data-path', type=str, required=True)
    cmd.add_argument('--embed-dir', type=str, required=True)
    cmd.add_argument('--model-dir', type=str, default='model/model/models--facebook--contriever-msmarco/snapshots/abe8c1493371369031bcb1e02acb754cf4e162fa')
    cmd.add_argument('--batch-size', type=int, default=200)
    args = cmd.parse_args(sys.argv[1:])

    generate_embeddings(args.data_path, args.embed_dir, args.model_dir, args.batch_size)