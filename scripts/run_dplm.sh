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
output_file=~/projects/stanfordnlp/tmp/$((RANDOM % 1000000)).dev.pred.conllu
gold_file=~/projects/stanfordnlp/data/dev.conllu
test_pred_file=~/projects/stanfordnlp/data/$((RANDOM % 1000000)).test.pred.conllu
test_file=~/projects/stanfordnlp/data/test.conllu

args=$@
lang=zh
short=zh_gsd
batch_size=1000
lm_batch_size=5000
clip=0.5
lambd=0.01
# --no_linearization \
    # --no_distance \

echo "Running parser with $args..."
dropout=0.4
wdecay=1e-6
lr=3e-3
python -u -m stanfordnlp.models.parser_lm --wordvec_dir $WORDVEC_DIR --train_file $train_file --eval_file $eval_file \
    --lm_file ~/projects/stanfordnlp/data/src_train.tsv ~/projects/stanfordnlp/data/trg_train.tsv \
    --output_file $output_file --gold_file $gold_file --lang $lang --shorthand $short --mode train \
    --batch_size $batch_size --lm_batch_size $lm_batch_size \
    --eval_interval 100 \
    --deep_biaff_hidden_dim 50 \
    --lambd $lambd \
    \
    --lstm_type wdlstm --lemma_emb_dim 0 --word_emb_dim 300 --char_emb_dim 0 --num_layers 2 --transformed_dim 0 --hidden_dim 1150 \
    \
    --max_grad_norm $clip \
    --vocab_cutoff 3 --rec_dropout 0.5 --word_dropout 0.1 \
    --lr $lr --wdecay $wdecay --dropout $dropout --seed $((RANDOM % 10000)) $args

# python -u -m stanfordnlp.models.parser --wordvec_dir $WORDVEC_DIR --eval_file $test_file \
#     --output_file $test_pred_file --gold_file $test_file --lang $lang --shorthand $short --mode predict $args
# results=`python stanfordnlp/utils/conll18_ud_eval.py -v $gold_file $output_file | head -12 | tail -n+12 | awk '{print $7}'`

# echo $results $args >> ${DEPPARSE_DATA_DIR}/${short}.results
# echo $short $results $args
