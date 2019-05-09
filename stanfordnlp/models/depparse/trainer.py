"""
A trainer class to handle training and testing of models.
"""

import sys
import torch
from torch import nn

from stanfordnlp.models.common.trainer import Trainer as BaseTrainer
from stanfordnlp.models.common import utils, loss
from stanfordnlp.models.common.chuliu_edmonds import chuliu_edmonds_one_root
from stanfordnlp.models.depparse.model import Parser
from stanfordnlp.models.pos.vocab import MultiVocab
from stanfordnlp.models.common.rnn_utils import copy_rnn_weights


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
            self.model = Parser(args, vocab, emb_matrix=pretrain.emb)
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if self.use_cuda:
            self.model.cuda()
        else:
            self.model.cpu()
        self.optimizer = utils.get_optimizer(self.args['optim'], self.parameters, self.args['lr'],
                                             betas=(0.9, self.args['beta2']), eps=1e-6, weight_decay=weight_decay)

    def update(self, batch, eval=False):
        inputs, orig_idx, word_orig_idx, sentlens, wordlens = unpack_batch(batch, self.use_cuda)
        word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel = inputs

        if eval:
            self.model.eval()
        else:
            self.model.train()
            self.optimizer.zero_grad()
        loss, _ = self.model(word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens)
        loss_val = loss.data.item()
        if eval:
            return loss_val

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
        self.optimizer.step()
        return loss_val

    def predict(self, batch, unsort=True):
        inputs, orig_idx, word_orig_idx, sentlens, wordlens = unpack_batch(batch, self.use_cuda)
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

    def init_from_lm(self, lm_model, freeze:bool=False):
        def freeze_net(net):
            for p in net.parameters():
                p.requires_grad = False

        param_list = ['word_emb', 'lemma_emb', 'upos_emb', 'xpos_emb',
                      'ufeats_emb', 'charmodel', 'trans_char', 'trans_char',
                      'trans_pretrained']
        lstm_params = ['parserlstm', 'lstm_forward', 'lstm_backward']

        for p_name in param_list:
            if hasattr(self.model, p_name):
                print('Initilizing {} from pretrained lm'.format(p_name))
                if not hasattr(lm_model, p_name):
                    raise ValueError('lm model does not have attribute {}'.format(p_name))
                module = getattr(lm_model, p_name)
                setattr(self.model, p_name, module)
                if freeze:
                    freeze_net(module)
            else:
                print('Module not found, skipping {}'.format(p_name))

        for p_name in lstm_params:
            if hasattr(self.model, p_name):
                print('Initilizing {} from pretrained lm'.format(p_name))
                if not hasattr(lm_model, p_name):
                    raise ValueError('lm model does not have attribute {}'.format(p_name))
                copy_rnn_weights(getattr(lm_model, p_name), getattr(self.model, p_name))
                if freeze:
                    freeze_net(getattr(self.model, p_name))
            else:
                print('Module not found, skipping {}'.format(p_name))

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
