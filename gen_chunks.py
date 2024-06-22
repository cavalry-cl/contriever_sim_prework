import torch
import os
import argparse
import sys
from fairseq.data.indexed_dataset import MMapIndexedDataset
from fairseq.data.indexed_dataset import MMapIndexedDatasetBuilder
import numpy as np


def generate_chunks(data_path, output_dir, chunk_len):
    dataset = MMapIndexedDataset(data_path)
    
    bin_path = output_dir + '.bin'
    index_path = output_dir + '.idx'
    mmap_out = MMapIndexedDatasetBuilder(bin_path)
    buff = torch.empty((chunk_len,))

    def flush():
        mmap_out.add_item(buff.clone())

    for (seg_id, seg) in enumerate(dataset):
        seg_len = seg.shape[0]
        q = seg_len // chunk_len
        r = seg_len % chunk_len
        for i in range(q):
            buff[:] = seg[i * chunk_len : (i+1) * chunk_len]
            flush()
        if r:
            buff[:r] = seg[q * chunk_len : ]
            buff[r:] = torch.zeros((seg_len - r,))
            flush()
    mmap_out.finalize(index_path)
    print('Chunking done!')


if __name__ == '__main__':
    cmd = argparse.ArgumentParser('Preprocessing-chunk_building arguments')
    cmd.add_argument('--data-path', type=str, required=True)
    cmd.add_argument('--output-dir', type=str, required=True)
    cmd.add_argument('--chunk-length', type=int, required=True)
    args = cmd.parse_args(sys.argv[1:])
    generate_chunks(args.data_path, args.output_dir, args.chunk_len)