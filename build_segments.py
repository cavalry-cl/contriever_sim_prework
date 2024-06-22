from fairseq.data.indexed_dataset import MMapIndexedDataset
from fairseq.data.indexed_dataset import MMapIndexedDatasetBuilder
import numpy as np
import argparse
import os
import sys
import torch


def build_segment(path, segment_len, tokenizer, mmap_output_path, json_output_path):
    dataset = MMapIndexedDataset(path)
    bin_path = f'{mmap_output_path}.bin'
    idx_path = f'{mmap_output_path}.idx'
    mmap_out = MMapIndexedDatasetBuilder(bin_path)
    total = len(dataset.sizes)

    offset = 0
    rest_len = segment_len
    buff = np.zeros(segment_len, dtype=int)
    segments = []
    segment_id = 0

    def flush():
        mmap_out.add_item(torch.tensor(buff))
        s = tokenizer.decode(np.abs(buff), skip_special_tokens=True)
        segments.append({"id": segment_id, 'contents': s})

    for sent_i in range(total):
        arr = dataset[sent_i]

        while len(arr) >= rest_len:
            buff[offset: offset + rest_len] = arr[:rest_len]
            flush()
            segment_id += 1
            offset = 0
            rest_len = segment_len
            arr = arr[rest_len:]

        rest_len -= len(arr)
        buff[offset: offset + len(arr)] = arr
        offset += len(arr)
    
    # drop the last incomplete item
    mmap_out.finalize(idx_path)
    print('Build dataset complete!')

    with open(json_output_path, 'w') as f:
        import json
        json.dump(segments, f)
    print('Build segment complete!')
    


if __name__ == '__main__':
    cmd = argparse.ArgumentParser('Segmenting corpus')
    cmd.add_argument('--data-path', type=str, required=True)
    cmd.add_argument('--vocab-dir', default=None, type=str)
    cmd.add_argument('--json-output-path', type=str, required=True)
    cmd.add_argument('--mmap-output-path', type=str, required=True)
    cmd.add_argument('--segment-length', type=int, default=1024)

    args = cmd.parse_args(sys.argv[1:])

    from transformers import AutoTokenizer
    tokenzier = AutoTokenizer.from_pretrained(args.vocab_dir)
    build_segment(args.data_path, args.segment_length, tokenzier, args.mmap_output_path, args.json_output_path)