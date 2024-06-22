import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import nltk
import argparse
import sys
import torch
from fairseq.data.indexed_dataset import MMapIndexedDatasetBuilder
from fairseq.data.indexed_dataset import MMapIndexedDataset
import json
import numpy as np
from tqdm import tqdm

PAD = 0

def retokenize_and_truncate(tokenizer, segment_len, input_dir, output_dir):
    data_file = f'{output_dir}.bin'
    index_file = f'{output_dir}.idx'
    mmap_builder = MMapIndexedDatasetBuilder(data_file)
    with open(input_dir, mode='r') as f_in:
        dics = json.load(f_in)
        for line in tqdm(dics):
            line = line['contents'].strip()
            line_id = torch.zeros((segment_len,),dtype=torch.int)
            offset = 0
            sents = nltk.sent_tokenize(line)
            for sent in sents:
                ids = tokenizer.encode(sent)
                ids[-1] = -ids[-1]
                cur_len = min(len(ids), segment_len - offset)
                line_id[offset : offset + cur_len] = torch.tensor(ids)[:cur_len]
                offset += cur_len
                if offset == segment_len:
                    break
            mmap_builder.add_item(line_id)

    mmap_builder.finalize(index_file)
    print('retokenizing and truncating done!')

if __name__ == '__main__':
    cmd = argparse.ArgumentParser('Preproceccing arguments')
    cmd.add_argument('--input-dir', type=str, required=True)
    cmd.add_argument('--output-dir', type=str, required=True)
    cmd.add_argument('--vocab-dir', type=str, required=True)
    cmd.add_argument('--segment-length', type=int, required=True)

    args = cmd.parse_args(sys.argv[1:])

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.vocab_dir)
    retokenize_and_truncate(tokenizer, args.segment_length, args.input_dir, args.output_dir)
    # truncate(tokenizer, args.segment_length, args.output_dir)