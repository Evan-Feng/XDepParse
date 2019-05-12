"""
A trainer class to handle training and testing of language models.
"""

import sys
import torch
from torch import nn

from stanfordnlp.models.common.trainer import Trainer as BaseTrainer
from stanfordnlp.models.common import utils, loss
from stanfordnlp.models.common.chuliu_edmonds import chuliu_edmonds_one_root
from stanfordnlp.models.lm.model import LSTMBiLM
from stanfordnlp.models.pos.vocab import MultiVocab


def unpack_batch(batch, use_cuda):
    """ Unpack a batch from the data loader. """
    if use_cuda:
        inputs = [b.cuda() if b is not None else None for b in batch[:11]]
    else:
        inputs = batch[:11]
    orig_idx = batch[11]
    word_orig_idx = batch[12]
    sentlens = batch[13]
    wordlens = batch[14]
    return inputs, orig_idx, word_orig_idx, sentlens, wordlens


class Trainer(BaseTrainer):
    """ A trainer for training models. """

    def __init__(self, args=None, vocab=None, pretrain=None, model_file=None, use_cuda=False):
        self.use_cuda = use_cuda
        if model_file is not None:
            # load everything from file
            self.load(pretrain, model_file)
        else:
            assert all(var is not None for var in [args, vocab, pretrain])
            # build model from scratch
            self.args = args
            self.vocab = vocab
            self.model = LSTMBiLM(args, vocab, emb_matrix=pretrain.emb)
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if self.use_cuda:
            self.model.cuda()
        else:
            self.model.cpu()

        self.optimizer = utils.get_optimizer(self.args['optim'], self.parameters, self.args['lr'],
                                             betas=(self.args['beta1'], self.args['beta2']),
                                             weight_decay=self.args['wdecay'])

    def update(self, batch, eval=False):
        inputs, orig_idx, word_orig_idx, sentlens, wordlens = unpack_batch(batch, self.use_cuda)
        word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, next_word, prev_word = inputs

        if eval:
            self.model.eval()
        else:
            self.model.train()
            self.optimizer.zero_grad()
        loss, preds = self.model(word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats,
                                 pretrained, lemma, next_word, prev_word, word_orig_idx, sentlens, wordlens)
        loss_val = loss.data.item()

        if eval:
            return loss_val

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
        self.optimizer.step()
        return loss_val

    def predict(self, batch, l_sample: int=10):
        inputs, orig_idx, word_orig_idx, sentlens, wordlens = unpack_batch(batch, self.use_cuda)
        word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, next_word, prev_word = inputs

        self.model.eval()
        bs = word.size(0)
        _, preds = self.model(word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained,
                              lemma, next_word, prev_word, word_orig_idx, sentlens, wordlens)
        preds = preds[0].argmax(-1)  # use forward predictions only
        pred_tokens = []
        for i in range(bs):
            pred_tokens.append([word[i, 0]] + preds[i][:sentlens[i]].tolist())
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
        self.model = LSTMBiLM(self.args, self.vocab, emb_matrix=pretrain.emb)
        self.model.load_state_dict(checkpoint['model'], strict=False)
