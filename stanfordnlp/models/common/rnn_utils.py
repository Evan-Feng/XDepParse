import torch


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


def copy_weights(src_module, trg_module):
    trg_params = dict(trg_module.named_parameters())
    for p_name, p in src_module.named_parameters():
        trg_params[p_name].data.copy_(p.data)


def freeze_net(module):
    for p in module.parameters():
        p.requires_grad = False


if __name__ == '__main__':
    x = torch.randn(100, 40, 200)
    sentlens = torch.randint(0, 41, (100,))
    rev_x = reverse_padded_sequence(x, sentlens, batch_first=True)
    for xx, rev_xx, l in zip(x, rev_x, sentlens):
        assert (xx[:l] == rev_xx[:l].flip([0])).all()

    x = torch.randn(40, 100, 200)
    sentlens = torch.randint(0, 41, (100,))
    rev_x = reverse_padded_sequence(x, sentlens, batch_first=False)
    for i, l in enumerate(sentlens):
        assert (x[:l, i, :][:l] == rev_x[:l, i, :].flip([0])).all()

    print('reverse_padded_sequence passed all tests')
