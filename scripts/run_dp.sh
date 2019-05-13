#!/bin/bash

WORDVEC_DIR=extern_data/word2vec/

args=$@
lang=zh
short=zh_gsd
batch_size=1000
dropout=0.4
rec_dropout=0.5
word_dropout=0.1
wdecay=1e-4
lr=1e-3

echo "Running parser with $args..."

python -u -m stanfordnlp.models.parser \
    --mode train \
    --lang $lang  --shorthand $short \
    --wordvec_dir $WORDVEC_DIR --train_file data/train.conllu --eval_file data/dev.conllu --gold_file data/dev.conllu \
    --output_file data/dev.pred.conllu  \
    --batch_size $batch_size \
    --eval_interval 200 \
    --deep_biaff_hidden_dim 100  \
    --deprel_loss 0 \
    --unfreeze_points 1500 2500 3500 \
    --lstm_type wdlstm  --num_layers 3 --hidden_dim 800 --output_hidden_dim 300\
    --lemma_emb_dim 0 --word_emb_dim 300 --transformed_dim 0 --char_emb_dim 0 \
    --lr $lr --wdecay $wdecay \
    --dropout $dropout --rec_dropout $rec_dropout --word_dropout $word_dropout \
    --seed $((RANDOM % 10000)) $args

python -u -m stanfordnlp.models.parser \
    --mode predict \
    --lang $lang  --shorthand $short \
    --wordvec_dir $WORDVEC_DIR --eval_file data/test.conllu --gold_file data/test.conllu \
    --output_file data/test.pred.conllu $args
