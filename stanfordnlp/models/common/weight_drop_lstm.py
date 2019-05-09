import torch
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


if __name__ == '__main__':
    x = torch.randn(100, 30, 300)
    lstm = WeightDropLSTM(300, 400, 2, weight_dropout=0.5, batch_first=True)
    outputs, _ = lstm(x)
    print(outputs.size())
