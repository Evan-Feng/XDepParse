import torch
import torch.nn as nn
from torch.nn import Parameter


def _weight_drop(module, weights, dropout):
    """
    Helper for `WeightDrop`.
    """

    for name_w in weights:
        w = getattr(module, name_w)
        module.register_parameter(name_w + '_raw', Parameter(w))

    original_module_forward = module.forward

    def forward(*args, **kwargs):
        for name_w in weights:
            raw_w = getattr(module, name_w + '_raw')
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

    def __init__(self, input_size, hidden_size, num_layers, output_size=None,
                 bidirectional=False, dropout=0, weight_dropout=0, batch_first=True):
        super().__init__()
        self.num_layers = num_layers
        if output_size is None:
            output_size = hidden_size
        self.rnns = [WeightDropLSTM(input_size if l == 0 else hidden_size,
                                    hidden_size if l != num_layers - 1 else output_size,
                                    1, bidirectional=bidirectional,
                                    batch_first=batch_first, weight_dropout=weight_dropout) for l in range(num_layers)]
        self.rnns = nn.ModuleList(self.rnns)
        self.hidden_dps = nn.ModuleList([RNNDropout(dropout) for l in range(num_layers)])
        self._set_weights()

    def _set_weights(self):
        w_names = ['weight_ih_l{}', 'weight_hh_l{}', 'bias_ih_l{}', 'bias_hh_l{}', 'weight_hh_l{}_raw']
        for l in range(self.num_layers):
            for w_name in w_names:
                w = getattr(self.rnns[l], w_name.format(0))
                setattr(self, w_name.format(l), w)

    def forward(self, inputs, hx=None):
        """
        inputs: tensor of shape (batch_size, seq_len, input_size)
        hx: List of List of tensors, each of shape (ndir, batch_size, hid_size)
            len(hx) should be num_layers and len(hx[0]) should be 2
        """
        new_h = []
        outputs = inputs
        for l, (rnn, hid_dp) in enumerate(zip(self.rnns, self.hidden_dps)):
            outputs, hid = rnn(outputs, hx[l] if hx else None)
            new_h.append(hid)
            if l != self.num_layers - 1:
                outputs = hid_dp(outputs)
        return outputs, new_h
