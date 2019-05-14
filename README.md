# XDepParse: Cross-Domain Dependency Parsing

[![Travis Status](https://travis-ci.com/stanfordnlp/stanfordnlp.svg?token=RPNzRzNDQRoq2x3J2juj&branch=master)](https://travis-ci.com/stanfordnlp/stanfordnlp)
[![PyPI version](https://img.shields.io/pypi/v/stanfordnlp.svg?colorB=blue)](https://pypi.org/project/stanfordnlp/)

## Dependencies
- Python >= 3.6
- PyTorch 1.1

## Datasets

All the data used for model training is placed under the directory "./data/":

```
.
├── README.md
└── data
    ├── train_src.tsv            (source domain unlabeled data for LM training)
    ├── train_trg.tsv            (target domain unlabeled data for LM training)
    ├── test_src.tsv             (source domain unlabeled data for LM validation)
    ├── test_trg.tsv             (target domain unlabeled data for LM validation)
    ├── train.conllu             (source domain labeled data for parser training)
    ├── dev.conullu              (target domain labeled data for parser validation)
    └── test.conullu             (target domain labeled data for parser testing)
```

By default the language model is trained on train_src.tsv and train_trg.tsv and the model that achieves the lowest perplexity on test_src.tsv and test_trg.tsv will be used for finetuning.

The source domain unlabeled data is sampled from the newswire domain of [Chinese Treebank 8.0](<https://catalog.ldc.upenn.edu/LDC2013T21>) (CTB8), while the target domain unlabeled data is sampled from all the other CTB8 domains.

## Usage

### Download Word Vectors

Run the following script to download word vectors and place them under the specified directory:

```bash
git clone https://github.com/Evan-Feng/XDepParse.git
cd XDepParse
bash scripts/download_vectors.sh extern_data/word2vec/
```

Alternatively, you can download only the Chinese word vectors:

```bash
bash scripts/download_zh_vec.sh extern_data/word2vec/
```

### Train a Language Model

Pretrain a cross-domain LSTM language model using the unlabeled data and save it to directory save_model/lm1/:

```bash
bash scripts/run_lm.sh --save_dir save_model/lm1/
```

The language model architecture we use is [AWD-LSTM](<https://openreview.net/forum?id=SyyGPP0TZ¬eId=SyyGPP0TZ>), which is a three-layer LSTM language model with weight dropout. By default the number of hidden units is 1150 and weight-tying is used, to change the hidden dimensionality and disable weight-tying, you can run the following command:

```bash
bash scripts/run_lm.sh --save_dir save_model/lm2/ --hidden_dim 400 --tie_softmax 0
```

### Finetune LM for Dependency Parsing

To build a dependency parser, we need to finetune the LM on labeled data:

```bash
bash scripts/run_dp.sh --pretrain_lm save_models/lm1/ --save_dir save_models/dp1/
```

Note that if you use a different architecture for LM pretraining, you need to explicitly specify those command line arguments so that the parser has the same architecture with the LM. For example, if you have used the second command for LM pretraining, you need to run:

```bash
bash scripts/run_dp.sh --pretrain_lm save_models/lm2/ --save_dir save_models/dp2/ --hidden_dim 400 --output_hidden_dim 400
```

During finetuning, the pretrained parameters are frozen at first and only the dependency scorer is trained. After the model has converged, all the parameters are jointly finetuned. For more details about the finetuning procedure, please refer to the paper [ULMFiT](<https://www.aclweb.org/anthology/P18-1031>).

## References

[1] Qi, Peng, et al. [Universal Dependency Parsing from Scratch.](<https://www.aclweb.org/anthology/K18-2#page=168>) *CoNLL 2018* (2018): 160.

[2] Howard, Jeremy, and Sebastian Ruder. [Universal Language Model Fine-tuning for Text Classification.](https://www.aclweb.org/anthology/P18-1031) *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. 2018.
