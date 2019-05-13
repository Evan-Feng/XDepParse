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
from stanfordnlp.models.common.rnn_utils import copy_weights, freeze_net


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
                                             betas=(self.args['beta1'], self.args['beta2']),
                                             weight_decay=self.args['wdecay'])

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

    def init_from_lm(self, lm_model, freeze: bool=True,
                     m_names=['word_emb', 'lemma_emb', 'upos_emb', 'xpos_emb',
                              'ufeats_emb', 'charmodel', 'trans_char', 'trans_char',
                              'trans_pretrained', 'lstm_forward', 'lstm_backward']):
        """
        Initialize the paramters from a pretrained langauge model.

        lm_model: LSTMBiLM object
        freeze: bool (optional)
            if True, the initialized paramters are freezed and will be from the optimizer's param group
        m_names: List[str] (optional)
            a list of module names to initialize
        """
        for m in m_names:
            if hasattr(self.model, m):
                print('Initilizing {} from pretrained lm'.format(m))
                if not hasattr(lm_model, m):
                    raise ValueError('pretrained language model does not have attribute {}'.format(m))
                module = getattr(self.model, m)
                copy_weights(getattr(lm_model, m), module)
                if freeze:
                    freeze_net(module)
            else:
                print('Module not found, skipping {}'.format(m))

        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = utils.get_optimizer(self.args['optim'], self.parameters, self.args['lr'],
                                             betas=(self.args['beta1'], self.args['beta2']),
                                             weight_decay=self.args['wdecay'])

    def unfreeze(self, layer_id, lr):
        finetune_params = []

        if self.args['lstm_type'] == 'awdlstm':
            p_names = ['weight_hh_l0', 'weight_hh_l0_raw', 'weight_ih_l0', 'bias_hh_l0', 'bias_ih_l0']
            for p_name in p_names:
                getattr(self.model.lstm_forward.rnns[layer_id], p_name).requires_grad = True
                getattr(self.model.lstm_backward.rnns[layer_id], p_name).requires_grad = True
                finetune_params.append(getattr(self.model.lstm_forward.rnns[layer_id], p_name))
                finetune_params.append(getattr(self.model.lstm_backward.rnns[layer_id], p_name))

        else:
            p_names = [s.format(layer_id) for s in ['weight_hh_l{}', 'weight_hh_l{}_raw', 'weight_ih_l{}', 'bias_hh_l{}', 'bias_ih_l{}']]
            for p_name in p_names:
                getattr(self.model.lstm_forward, p_name).requires_grad = True
                getattr(self.model.lstm_backward, p_name).requires_grad = True
                finetune_params.append(getattr(self.model.lstm_forward, p_name))
                finetune_params.append(getattr(self.model.lstm_backward, p_name))
        # print(self.optimizer.param_groups)
        print()
        print('Adding params {} with lr {}'.format(p_names, lr))
        print()
        self.optimizer.add_param_group({
            'params': finetune_params,
            'lr': lr,
        })
        # print(self.optimizer.param_groups)

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
