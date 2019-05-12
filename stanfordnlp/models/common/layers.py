import torch
import torch.nn as nn
from torch.nn import Parameter


def _weight_drop(module, weights, dropout):
    """
    Helper for `WeightDrop`.
    """

    for name_w in weights:
        w = getattr(module, name_w)
        # del module._parameters[name_w]
        module.register_parameter(name_w + '_raw', Parameter(w))

    original_module_forward = module.forward

    def forward(*args, **kwargs):
        for name_w in weights:
            raw_w = getattr(module, name_w + '_raw')
            # w = torch.nn.functional.dropout(raw_w, p=dropout, training=module.training)
            # setattr(module, name_w, w)
            module._parameters[name_w] = torch.nn.functional.dropout(raw_w, p=dropout, training=module.training)

        return original_module_forward(*args)

    setattr(module, 'forward', forward)


class WeightDropLSTM(torch.nn.LSTM):
    """
    Wrapper around :class:`torch.nn.LSTM` that adds ``weight_dropout`` named argument.

    Args:
        weight_dropout (float): The probability a weight will be dropped.
    """

    def __init__(self, *args, weight_dropout=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        weights = ['weight_hh_l' + str(i) for i in range(self.num_layers)]
        _weight_drop(self, weights, weight_dropout)


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

    def __init__(self, input_size, hidden_size, output_size, num_layers,
                 bidirectional=False, dropout=0, weight_dropout=0, batch_first=True):
        super().__init__()
        self.num_layers = num_layers
        self.rnns = [WeightDropLSTM(input_size if l == 0 else hidden_size,
                                    hidden_size if l != num_layers - 1 else output_size,
                                    1, bidirectional=bidirectional,
                                    batch_first=batch_first, weight_dropout=weight_dropout) for l in range(num_layers)]
        self.rnns = nn.ModuleList(self.rnns)
        self.hidden_dps = nn.ModuleList([RNNDropout(dropout) for l in range(num_layers)])

    def forward(self, inputs):
        raw_output = inputs
        for l, (rnn, hid_dp) in enumerate(zip(self.rnns, self.hidden_dps)):
            raw_output, _ = rnn(raw_output)
            if l != self.num_layers - 1:
                raw_output = hid_dp(raw_output)
        return raw_output, None
