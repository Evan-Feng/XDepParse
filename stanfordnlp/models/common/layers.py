import torch
import torch.nn as nn
from stanfordnlp.models.common.weight_drop_lstm import WeightDropLSTM


def dropout_mask(x, sz, p: float):
    """
    Return a dropout mask of the same type as `x`, size `sz`, with probability `p` to cancel an element.
    """
    return x.new(*sz).bernoulli_(1 - p).div_(1 - p)


class RNNDropout(nn.Module):
    """
    Dropout with probability `p` that is consistent on the seq_len dimension.
    """

    def __init__(self, p: float=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.:
            return x
        m = dropout_mask(x.data, (x.size(0), 1, x.size(2)), self.p)
        return x * m


class MultiLayerLSTM(nn.Module):
    """
    A multi-layer LSTM with specified output dimension.
    """

    def __init__(self, in_dim, hid_dim, out_dim, n_layers, bidirectional=False, dropout=0, weight_dropout=0):
        super().__init__()
        self.n_layers = n_layers
        self.rnns = [WeightDropLSTM(in_dim if l == 0 else hid_dim,
                                    hid_dim if l != n_layers - 1 else out_dim,
                                    1, bidirectional=bidirectional,
                                    batch_first=True, weight_dropout=weight_dropout) for l in range(n_layers)]
        self.rnns = nn.ModuleList(self.rnns)
        self.hidden_dps = nn.ModuleList([RNNDropout(dropout) for l in range(n_layers)])

    def forward(self, inputs):
        raw_output = inputs
        for l, (rnn, hid_dp) in enumerate(zip(self.rnns, self.hidden_dps)):
            raw_output, _ = rnn(raw_output)
            if l != self.n_layers - 1:
                raw_output = hid_dp(raw_output)
        return raw_output, None
