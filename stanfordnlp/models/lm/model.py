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
from stanfordnlp.models.common.layers import WeightDropLSTM
from stanfordnlp.models.common.rnn_utils import reverse_padded_sequence
from stanfordnlp.models.common.layers import MultiLayerLSTM


class LSTMBiLM(nn.Module):

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
        rnn_params = {
            'input_size': input_size,
            'hidden_size': self.args['hidden_dim'],
            'num_layers': self.args['num_layers'],
            'batch_first': True,
            'bidirectional': False,
            'dropout': self.args['dropout'],
            ('rec_dropout' if self.args['lstm_type'] == 'hlstm' else 'weight_dropout'): self.args['rec_dropout'],
        }
        if args['lstm_type'] == 'hlstm':
            self.lstm_forward = HighwayLSTM(**rnn_params, highway_func=torch.tanh)
            self.lstm_backward = HighwayLSTM(**rnn_params, highway_func=torch.tanh)
        elif args['lstm_type'] == 'wdlstm':
            out_dim = args['word_emb_dim'] if args['tie_softmax'] else args['hidden_dim']
            self.lstm_forward = MultiLayerLSTM(**rnn_params, output_size=out_dim)
            self.lstm_backward = MultiLayerLSTM(**rnn_params, output_size=out_dim)
        else:
            raise ValueError('LSTM type not supported')

        self.drop_replacement = nn.Parameter(torch.randn(input_size) / np.sqrt(input_size))
        # self.parserlstm_h_init = nn.Parameter(torch.zeros(2 * self.args['num_layers'], 1, self.args['hidden_dim']))
        # self.parserlstm_c_init = nn.Parameter(torch.zeros(2 * self.args['num_layers'], 1, self.args['hidden_dim']))

        if args['tie_softmax']:
            self.dec_forward = nn.Linear(self.args['word_emb_dim'], len(vocab['word']))
            self.dec_backward = nn.Linear(self.args['word_emb_dim'], len(vocab['word']))
            self.dec_forward.weight = self.word_emb.weight
            self.dec_backward.weight = self.word_emb.weight

        else:
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

        # lstm_inputs = PackedSequence(lstm_inputs, inputs[0].batch_sizes)
        # rev_lstm_inputs = PackedSequence(rev_lstm_inputs, inputs[0].batch_sizes)

        # forward / backward language model
        if self.args['lstm_type'] == 'hlstm':
            lstm_inputs = pack(lstm_inputs)
            rev_lstm_inputs = pack(rev_lstm_inputs)
            hid_forward, _ = self.lstm_forward(lstm_inputs, sentlens)
            hid_backward, _ = self.lstm_backward(rev_lstm_inputs, sentlens)
            hid_forward, _ = pad_packed_sequence(hid_forward, batch_first=True)
            hid_backward, _ = pad_packed_sequence(hid_backward, batch_first=True)
        elif self.args['lstm_type'] in ('wdlstm', 'awdlstm'):
            hid_forward, _ = self.lstm_forward(lstm_inputs)
            hid_backward, _ = self.lstm_backward(rev_lstm_inputs)

        hid_backward = reverse_padded_sequence(hid_backward, sentlens, batch_first=True)

        scores_forward = self.dec_forward(self.drop(hid_forward))
        scores_backward = self.dec_backward(self.drop(hid_backward))

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
