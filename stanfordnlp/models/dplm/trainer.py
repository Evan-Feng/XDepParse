"""
A trainer class to handle training and testing of models.
"""

import sys
import torch
from torch import nn
import numpy as np

from stanfordnlp.models.common.trainer import Trainer as BaseTrainer
from stanfordnlp.models.common import utils, loss
from stanfordnlp.models.common.chuliu_edmonds import chuliu_edmonds_one_root
from stanfordnlp.models.dplm.model import ParserLM
from stanfordnlp.models.pos.vocab import MultiVocab
from stanfordnlp.models.common.rnn_utils import copy_rnn_weights
from stanfordnlp.models.depparse.trainer import unpack_batch as depparse_unpack
from stanfordnlp.models.lm.trainer import unpack_batch as lm_unpack


class Trainer(BaseTrainer):
    """ A trainer for training models. """

    def __init__(self, args=None, vocab=None, pretrain=None, model_file=None, use_cuda=False, weight_decay=0):
        self.use_cuda = use_cuda
        if model_file is not None:
            # load everything from file
            self.load(pretrain, model_file)
        else:
            assert all(var is not None for var in [args, vocab, pretrain])
            # build model from scratch
            self.args = args
            self.vocab = vocab
            self.model = ParserLM(args, vocab, emb_matrix=pretrain.emb)
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if self.use_cuda:
            self.model.cuda()
        else:
            self.model.cpu()
        # self.optimizer = utils.get_optimizer(self.args['optim'], self.parameters, self.args['lr'],
        #                                      betas=(0.9, self.args['beta2']), eps=1e-6, weight_decay=weight_decay)
        if self.args['optim'] == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args['lr'], weight_decay=self.args['wdecay'])
        else:
            raise NotImplementedError()

    def update(self, dp_batch, lm_batch, eval=False):
        inputs, orig_idx, word_orig_idx, sentlens, wordlens = depparse_unpack(dp_batch, self.use_cuda)
        word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel = inputs

        if eval:
            self.model.eval()
        else:
            self.model.train()
            self.optimizer.zero_grad()

        depparse_loss, _ = self.model(word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats,
                                      pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens)

        inputs, orig_idx, word_orig_idx, sentlens, wordlens = lm_unpack(lm_batch, self.use_cuda)
        word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, next_word, prev_word = inputs

        lm_loss, _ = self.model.lm_forward(word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats,
                                           pretrained, lemma, next_word, prev_word, word_orig_idx, sentlens, wordlens)
        loss = depparse_loss * self.args['lambd'] + lm_loss
        # print(np.exp(lm_loss.item()))

        if not eval:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
            self.optimizer.step()

        depparse_loss = depparse_loss.item()
        lm_loss = lm_loss.item()
        loss = loss.item()

        return depparse_loss, lm_loss, loss

    def predict(self, batch, unsort=True):
        inputs, orig_idx, word_orig_idx, sentlens, wordlens = depparse_unpack(batch, self.use_cuda)
        word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel = inputs

        self.model.eval()
        batch_size = word.size(0)
        _, preds = self.model(word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens)
        head_seqs = [chuliu_edmonds_one_root(adj[:l, :l])[1:] for adj, l in zip(preds[0], sentlens)]  # remove attachment for the root
        deprel_seqs = [self.vocab['deprel'].unmap([preds[1][i][j + 1][h] for j, h in enumerate(hs)]) for i, hs in enumerate(head_seqs)]

        pred_tokens = [[[str(head_seqs[i][j]), deprel_seqs[i][j]] for j in range(sentlens[i] - 1)] for i in range(batch_size)]
        if unsort:
            pred_tokens = utils.unsort(pred_tokens, orig_idx)
        return pred_tokens

    def save(self, filename, skip_modules=True):
        model_state = self.model.state_dict()
        # skip saving modules like pretrained embeddings, because they are large and will be saved in a separate file
        if skip_modules:
            skipped = [k for k in model_state.keys() if k.split('.')[0] in self.model.unsaved_modules]
            for k in skipped:
                del model_state[k]
        params = {
            'model': model_state,
            'vocab': self.vocab.state_dict(),
            'config': self.args
        }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def load(self, pretrain, filename):
        try:
            checkpoint = torch.load(filename, lambda storage, loc: storage)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            sys.exit(1)
        self.args = checkpoint['config']
        self.vocab = MultiVocab.load_state_dict(checkpoint['vocab'])
        self.model = Parser(self.args, self.vocab, emb_matrix=pretrain.emb)
        self.model.load_state_dict(checkpoint['model'], strict=False)
