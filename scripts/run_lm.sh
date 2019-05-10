#!/bin/bash
#
# Train and evaluate parser. Run as:
#   ./run_depparse.sh TREEBANK TAG_TYPE OTHER_ARGS
# where TREEBANK is the UD treebank name (e.g., UD_English-EWT) and OTHER_ARGS are additional training arguments (see parser code) or empty.
# This script assumes UDBASE and DEPPARSE_DATA_DIR are correctly set in config.sh.

source scripts/config.sh


WORDVEC_DIR=~/projects/stanfordnlp/extern_data/word2vec/

train_file=~/projects/stanfordnlp/data/train.conllu
eval_file=~/projects/stanfordnlp/data/dev.conllu
args=$@
lang=zh
short=zh_gsd
batch_size=5000

echo "Running language model with $args..."
dropout=0.4
word_dropout=0.1
wdecay=1e-6
lr=3e-3
rec_dropout=0.5
clip=0.25

python -u -m stanfordnlp.models.language_model --train_file ~/projects/stanfordnlp/data/src_train.tsv ~/projects/stanfordnlp/data/trg_train.tsv \
    --eval_file ~/projects/stanfordnlp/data/src_test.tsv ~/projects/stanfordnlp/data/trg_test.tsv \
    --lang $lang --shorthand $short --batch_size $batch_size --mode train --eval_interval 400 \
    \
    --lstm_type awdlstm --lemma_emb_dim 0 --word_emb_dim 300 --transformed_dim 0 --char_emb_dim 0 --num_layers 2 --hidden_dim 1150 \
    \
    --lr $lr \
    --wdecay $wdecay --dropout $dropout --rec_dropout $rec_dropout --word_dropout $word_dropout \
    --max_grad_norm $clip --seed $((RANDOM % 10000)) $args

# results=`python stanfordnlp/utils/conll18_ud_eval.py -v $gold_file $output_file | head -12 | tail -n+12 | awk '{print $7}'`

# echo $results $args >> ${DEPPARSE_DATA_DIR}/${short}.results
# echo $short $results $args
