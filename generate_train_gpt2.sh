python corpus_preprocess.py \
    --corpus-type wikitext103 \
    --dir corpus \
    --vocab-dir gpt2 \
    --output-dir corpus/gpt2-train/tokenized

python build_segments.py \
    --data-path corpus/gpt2-train/tokenized/wiki.train \
    --vocab-dir gpt2 \
    --segment-length 512 \
    --json-output-path corpus/gpt2-train/gpt2-train.segments.json \
    --mmap-output-path corpus/gpt2-train

python retokenize_and_truncate.py \
    --input-dir corpus/gpt2-train/gpt2-train.segments.json \
    --vocab-dir facebook/contriever-msmarco \
    --segment-length 512 \
    --output-dir corpus/retokenized-train

python gen_embed.py \
    --data-path corpus/retokenized-train \
        --model-dir facebook/contriever-msmarco \
    --embed-dir corpus/gpt2-train/embedded/train 

python build_knn_brute.py \
    --embed-path corpus/gpt2-train/embedded/train_embed.npy \
    --knn-path knns/gpt2-train/

python sort.py \
    --embed-path corpus/gpt2-train/embedded/train_embed.npy \
    --knn-path knns/gpt2-train/ \
    --sort-path gpt2-train_sort.json