import torch
import numpy as np
from tqdm import *
import os
import time
import json


if torch.cuda.is_available():
	device = torch.device("cuda")
else:
	device = torch.device("cpu")

def get_knn(doc_id, batch_size, knn_path):
    batch_id = doc_id // batch_size
    knns = np.load(knn_path + f'batch{batch_id}.npy')
    return knns[doc_id % batch_size]

def sort(k, batch_size, embed_path, knn_path, sort_path):

    embed = np.load(embed_path)
    dataset_size = embed.shape[0]

    k = min(k, dataset_size)
    unseen_docs = set(range(dataset_size))
    clusters = [[] for _ in range(dataset_size)]
    cluster_size = [0 for _ in range(dataset_size)]
    cluster_id = [0 for _ in range(dataset_size)]
    cluster_cnt = 0
    min_cluster_size = k

    # cur_doc = unseen_docs.pop()
    unseen_docs.remove(15000)
    cur_doc = 15000

    clusters[cluster_cnt].append(cur_doc)
    cluster_size[cluster_cnt] += 1
    cluster_id[cur_doc] = cluster_cnt

    def find_first_unseen_doc(knns):
        for doc_id in range(k):
            if knns[doc_id] in unseen_docs:
                return knns[doc_id]
        return None

    with tqdm(total=dataset_size-1) as pbar:
        while unseen_docs:
            knns = get_knn(cur_doc, batch_size, knn_path)
            next_doc = find_first_unseen_doc(knns)
            # if next_doc is None or cluster_size[cur_doc] > min_cluster_size:
            if next_doc is None:
                cur_doc = unseen_docs.pop()
                cluster_cnt += 1
            else:
                cur_doc = next_doc
                unseen_docs.remove(cur_doc)
            clusters[cluster_cnt].append(cur_doc)
            cluster_id[cur_doc] = cluster_cnt
            cluster_size[cluster_cnt] += 1
            pbar.update(1)

    cluster_cnt += 1
    print(f'cluster_cnt={cluster_cnt}')


    def first_doc_knn_not_in_the_cluster(knns, cluster):
        for i in range(k):
            doc_id = knns[i]
            if cluster_id[doc_id] != cluster and cluster_id[doc_id] != -1:
                return doc_id, cluster_id[doc_id]
        return None, None

    cnt = 0
    res = []
    for cluster in trange(cluster_cnt):
        cluster_dic = {}
        cluster_dic['cluster_id'] = cnt
        cluster_dic['doc_seq'] = list(map(int,clusters[cluster]))
        res.append(cluster_dic)
        cnt += 1

    with open('vanilla'+sort_path, 'w') as f:
        json.dump(res, f)


    deleted = [0 for _ in range(cluster_cnt)]
    merged_clusters_num = 0
    unmatched = []
    for cluster in trange(cluster_cnt):
        cluster_docs = clusters[cluster]
        if len(cluster_docs) < min_cluster_size:
            merged_clusters_num += 1
            # print(merged_clusters_num)
            for doc in cluster_docs:
                top1k, top1k_cluster = first_doc_knn_not_in_the_cluster(get_knn(doc, batch_size, knn_path), cluster)
                assert(cluster_id[top1k] == top1k_cluster)
                assert(top1k in clusters[top1k_cluster])
                if top1k == None:
                    unmatched.append(doc)
                    cluster_id[doc] = -1
                    continue
                k_cluster_docs = clusters[top1k_cluster]
                # add k to doc
                k_cluster_docs.insert(k_cluster_docs.index(top1k), doc)
                cluster_id[doc] = top1k_cluster

                # update the cluster
                clusters[top1k_cluster] = k_cluster_docs
            # del clusters[cluster]
            deleted[cluster] = 1
            cluster -= 1
    print(f'merged_clusters_num={merged_clusters_num}')

    print(f'unmatched_cnt={len(unmatched)}')

    merged_clusters_id = 0
    res = []
    for cluster in trange(cluster_cnt):
        if deleted[cluster]:
            continue
        cluster_dic = {}
        cluster_dic['cluster_id'] = merged_clusters_id
        cluster_dic['doc_seq'] = list(map(int,clusters[cluster]))
        merged_clusters_id += 1
        res.append(cluster_dic)

    with open(sort_path, 'w') as f:
        json.dump(res, f)


import sys
import argparse

if __name__ == '__main__':
    cmd = argparse.ArgumentParser('Preprocessing-embed_building arguments')
    cmd.add_argument('--k', type=int, default=1024)
    cmd.add_argument('--batch-size', type=int, default=1000)
    cmd.add_argument('--embed-path', type=str)
    cmd.add_argument('--knn-path', type=str)
    cmd.add_argument('--sort-path', type=str)
    args = cmd.parse_args(sys.argv[1:])

    sort(args.k, args.batch_size, args.embed_path, args.knn_path, args.sort_path)
