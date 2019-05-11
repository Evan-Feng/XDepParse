#!/bin/bash

WORDVEC_DIR=extern_data/word2vec/

args=$@
lang=zh
short=zh_gsd
batch_size=5000
dropout=0.4
word_dropout=0.1
wdecay=1e-6
lr=3e-3
rec_dropout=0.5
clip=0.25

echo "Running language model with $args..."

python -u -m stanfordnlp.models.language_model \
    --mode train \
    --lang $lang --shorthand $short \
    --train_file data/train_src.tsv data/train_trg.tsv \
    --eval_file data/test_src.tsv data/test_trg.tsv \
    --wordvec_dir $WORDVEC_DIR \
    --batch_size $batch_size --mode train --eval_interval 400 \
    --lstm_type awdlstm --num_layers 3 --hidden_dim 1150 \
    --lemma_emb_dim 0 --word_emb_dim 300 --transformed_dim 0 --char_emb_dim 0  \
    --lr $lr --wdecay $wdecay \
    --dropout $dropout --rec_dropout $rec_dropout --word_dropout $word_dropout \
    --max_grad_norm $clip \
    --seed $((RANDOM % 10000)) $args
