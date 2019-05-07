import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pack_sequence, PackedSequence

from stanfordnlp.models.common.vocab import PAD_ID
from stanfordnlp.models.common.biaffine import DeepBiaffineScorer
from stanfordnlp.models.common.hlstm import HighwayLSTM
from stanfordnlp.models.common.dropout import WordDropout
from stanfordnlp.models.common.vocab import CompositeVocab
from stanfordnlp.models.common.char_model import CharacterModel


def reverse_padded_sequence(inputs, lengths, batch_first=False):
    """Reverses sequences according to their lengths.
    Inputs should have size ``T x B x *`` if ``batch_first`` is False, or
    ``B x T x *`` if True. T is the length of the longest sequence (or larger),
    B is the batch size, and * is any number of dimensions (including 0).
    Arguments:
        inputs (Variable): padded batch of variable length sequences.
        lengths (list[int]): list of sequence lengths
        batch_first (bool, optional): if True, inputs should be B x T x *.
    Returns:
        A Variable with the same size as inputs, but with each sequence
        reversed according to its length.
    """
    if batch_first:
        inputs = inputs.transpose(0, 1)
    max_length, batch_size = inputs.size(0), inputs.size(1)
    if len(lengths) != batch_size:
        raise ValueError('inputs is incompatible with lengths.')
    ind = [list(reversed(range(0, length))) + list(range(length, max_length))
           for length in lengths]
    ind = torch.LongTensor(ind).transpose(0, 1)
    for dim in range(2, inputs.dim()):
        ind = ind.unsqueeze(dim)
    ind = ind.expand_as(inputs)
    if inputs.is_cuda:
        ind = ind.cuda(inputs.get_device())
    reversed_inputs = torch.gather(inputs, 0, ind)
    if batch_first:
        reversed_inputs = reversed_inputs.transpose(0, 1)
    return reversed_inputs


class HLSTMLanguageModel(nn.Module):

    def __init__(self, args, vocab, emb_matrix=None, share_hid=False):
        super().__init__()

        self.vocab = vocab
        self.args = args
        self.share_hid = share_hid
        self.unsaved_modules = []

        def add_unsaved_module(name, module):
            self.unsaved_modules += [name]
            setattr(self, name, module)

        # input layers
        input_size = 0
        if self.args['word_emb_dim'] > 0:
            # frequent word embeddings
            self.word_emb = nn.Embedding(len(vocab['word']), self.args['word_emb_dim'], padding_idx=0)
            input_size += self.args['word_emb_dim']

        if self.args['lemma_emb_dim'] > 0:
            self.lemma_emb = nn.Embedding(len(vocab['lemma']), self.args['lemma_emb_dim'], padding_idx=0)
            input_size += self.args['lemma_emb_dim']

        if self.args['tag_emb_dim'] > 0:
            self.upos_emb = nn.Embedding(len(vocab['upos']), self.args['tag_emb_dim'], padding_idx=0)

            if not isinstance(vocab['xpos'], CompositeVocab):
                self.xpos_emb = nn.Embedding(len(vocab['xpos']), self.args['tag_emb_dim'], padding_idx=0)
            else:
                self.xpos_emb = nn.ModuleList()

                for l in vocab['xpos'].lens():
                    self.xpos_emb.append(nn.Embedding(l, self.args['tag_emb_dim'], padding_idx=0))

            self.ufeats_emb = nn.ModuleList()

            for l in vocab['feats'].lens():
                self.ufeats_emb.append(nn.Embedding(l, self.args['tag_emb_dim'], padding_idx=0))

            input_size += self.args['tag_emb_dim'] * 2

        if self.args['char'] and self.args['char_emb_dim'] > 0:
            self.charmodel = CharacterModel(args, vocab)
            self.trans_char = nn.Linear(self.args['char_hidden_dim'], self.args['transformed_dim'], bias=False)
            input_size += self.args['transformed_dim']

        if self.args['pretrain']:
            # pretrained embeddings, by default this won't be saved into model file
            add_unsaved_module('pretrained_emb', nn.Embedding.from_pretrained(torch.from_numpy(emb_matrix), freeze=True))
            self.trans_pretrained = nn.Linear(emb_matrix.shape[1], self.args['transformed_dim'], bias=False)
            input_size += self.args['transformed_dim']

        # recurrent layers
        self.lstm_forward = HighwayLSTM(input_size, self.args['hidden_dim'], self.args['num_layers'], batch_first=True, bidirectional=False,
                                        dropout=self.args['dropout'], rec_dropout=self.args['rec_dropout'], highway_func=torch.tanh)
        self.lstm_backward = HighwayLSTM(input_size, self.args['hidden_dim'], self.args['num_layers'], batch_first=True, bidirectional=False,
                                         dropout=self.args['dropout'], rec_dropout=self.args['rec_dropout'], highway_func=torch.tanh)
        self.drop_replacement = nn.Parameter(torch.randn(input_size) / np.sqrt(input_size))
        # self.parserlstm_h_init = nn.Parameter(torch.zeros(2 * self.args['num_layers'], 1, self.args['hidden_dim']))
        # self.parserlstm_c_init = nn.Parameter(torch.zeros(2 * self.args['num_layers'], 1, self.args['hidden_dim']))

        self.dec_forward = nn.Linear(self.args['hidden_dim'], len(vocab['word']))
        self.dec_backward = nn.Linear(self.args['hidden_dim'], len(vocab['word']))

        # criterion
        self.crit = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')  # ignore padding

        self.drop = nn.Dropout(args['dropout'])
        self.worddrop = WordDropout(args['word_dropout'])

    def forward(self, word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, next_word, prev_word, word_orig_idx, sentlens, wordlens):
        """
        Note: the original parameters `head` and `deprel` in dependency
        parser are replaced with `next_word` and `prev_word`
        """
        def pack(x):
            return pack_padded_sequence(x, sentlens, batch_first=True)

        inputs = []
        if self.args['pretrain']:
            pretrained_emb = self.pretrained_emb(pretrained)
            pretrained_emb = self.trans_pretrained(pretrained_emb)
            # pretrained_emb = pack(pretrained_emb)
            inputs += [pretrained_emb]

        # def pad(x):
        #    return pad_packed_sequence(PackedSequence(x, pretrained_emb.batch_sizes), batch_first=True)[0]

        if self.args['word_emb_dim'] > 0:
            word_emb = self.word_emb(word)
            # word_emb = pack(word_emb)
            inputs += [word_emb]

        if self.args['lemma_emb_dim'] > 0:
            lemma_emb = self.lemma_emb(lemma)
            # lemma_emb = pack(lemma_emb)
            inputs += [lemma_emb]

        if self.args['tag_emb_dim'] > 0:
            pos_emb = self.upos_emb(upos)

            if isinstance(self.vocab['xpos'], CompositeVocab):
                for i in range(len(self.vocab['xpos'])):
                    pos_emb = pos_emb + self.xpos_emb[i](xpos[:, :, i])
            else:
                pos_emb += self.xpos_emb(xpos)
            # pos_emb = pack(pos_emb)

            feats_emb = 0
            for i in range(len(self.vocab['feats'])):
                feats_emb = feats_emb + self.ufeats_emb[i](ufeats[:, :, i])
            # feats_emb = pack(feats_emb)

            inputs += [pos_emb, feats_emb]

        if self.args['char'] and self.args['char_emb_dim'] > 0:
            char_reps = self.charmodel(wordchars, wordchars_mask, word_orig_idx, sentlens, wordlens)
            # char_reps = PackedSequence(self.trans_char(self.drop(char_reps.data)), char_reps.batch_sizes)
            char_reps = self.trans_char(self.drop(char_reps.data))
            inputs += [char_reps]

        # lstm_inputs = torch.cat([x.data for x in inputs], -1)
        lstm_inputs = torch.cat(inputs, -1)

        lstm_inputs = self.worddrop(lstm_inputs, self.drop_replacement)
        lstm_inputs = self.drop(lstm_inputs)

        rev_lstm_inputs = reverse_padded_sequence(lstm_inputs, sentlens, batch_first=True)

        lstm_inputs = pack(lstm_inputs)
        rev_lstm_inputs = pack(rev_lstm_inputs)
        # lstm_inputs = PackedSequence(lstm_inputs, inputs[0].batch_sizes)
        # rev_lstm_inputs = PackedSequence(rev_lstm_inputs, inputs[0].batch_sizes)

        # forward language model
        hid_forward, _ = self.lstm_forward(lstm_inputs, sentlens)
        hid_forward, _ = pad_packed_sequence(hid_forward, batch_first=True)

        # backward language model
        hid_backward, _ = self.lstm_backward(rev_lstm_inputs, sentlens)
        hid_backward, _ = pad_packed_sequence(hid_backward, batch_first=True)
        hid_backward = reverse_padded_sequence(hid_backward, sentlens, batch_first=True)

        scores_forward = self.dec_forward(hid_forward)
        scores_backward = self.dec_backward(hid_backward)

        next_word = next_word.masked_fill(next_word == PAD_ID, -1)
        prev_word = prev_word.masked_fill(prev_word == PAD_ID, -1)

        preds = []

        if self.training:
            loss_forward = self.crit(scores_forward.view(-1, scores_forward.size(-1)), next_word.view(-1))
            loss_backward = self.crit(scores_backward.view(-1, scores_backward.size(-1)), prev_word.view(-1))
            # print('forward ppl {:.6f} {:.6f}'.format(np.exp(loss_forward.item() / wordchars.size(0)),
                                                     # np.exp(loss_backward.item() / wordchars.size(0))))
            loss = (loss_forward + loss_backward) / (2 * wordchars.size(0))  # number of words
            preds.append(F.log_softmax(scores_forward, -1).detach().cpu().numpy())
            preds.append(F.log_softmax(scores_backward, -1).detach().cpu().numpy())
        else:
            loss_forward = self.crit(scores_forward.view(-1, scores_forward.size(-1)), next_word.view(-1))
            loss_backward = self.crit(scores_backward.view(-1, scores_backward.size(-1)), prev_word.view(-1))
            loss = (loss_forward + loss_backward) / (2 * wordchars.size(0))  # number of words
            preds.append(F.log_softmax(scores_forward, -1).detach().cpu().numpy())
            preds.append(F.log_softmax(scores_backward, -1).detach().cpu().numpy())

        return loss, preds
