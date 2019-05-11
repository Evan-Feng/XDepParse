import random
import torch

from stanfordnlp.models.common.data import map_to_ids, get_long_tensor, get_float_tensor, sort_all
from stanfordnlp.models.common import conll
from stanfordnlp.models.common.vocab import PAD_ID, VOCAB_PREFIX, ROOT_ID, CompositeVocab
from stanfordnlp.models.pos.vocab import CharVocab, WordVocab, XPOSVocab, FeatureVocab, MultiVocab
from stanfordnlp.models.pos.xpos_vocab_factory import xpos_vocab_factory
from stanfordnlp.pipeline.doc import Document


class DataLoader:

    def __init__(self, input_srcs, batch_size, args, pretrain, vocab=None, evaluation=False, cutoff=7):
        """
        Note: input_srcs should be a list of strings instead of a single string
        """
        self.batch_size = batch_size
        self.args = args
        self.eval = evaluation
        self.shuffled = not self.eval

        # added
        self.cutoff = cutoff

        # check if input source is a file or a Document object
        # if isinstance(input_src, str):
        #     filename = input_src
        #     assert filename.endswith('tsv'), "Loaded file must be tsv file."
        #     data = self.load_file(filename, evaluation=self.eval)
        # elif isinstance(input_src, Document):
        #     raise NotImplementedError()

        data_list = [self.load_file(file) for file in input_srcs]
        for f_name, data in zip(input_srcs, data_list):
            print('File {} contains {} examples'.format(f_name, len(data)))

        # handle vocab
        if vocab is None:
            self.vocab = self.init_vocab(data_list)
        else:
            self.vocab = vocab
        self.pretrain_vocab = pretrain.vocab

        # filter and sample data
        if args.get('sample_train', 1.0) < 1.0 and not self.eval:
            raise NotImplementedError()

        if args['balance'] and not evaluation:
            print('Balancing data across files')
            l_max = max([len(data) for data in data_list])
            rates = [l_max // len(data) for data in data_list]
            ans = []
            for r, data in zip(rates, data_list):
                ans += data * r
            data = ans
        else:
            data = sum(data_list, [])

        data = self.preprocess(data, self.vocab, self.pretrain_vocab, args)
        # shuffle for training
        if self.shuffled:
            print('Shuffling data')
            random.shuffle(data)

        # def sum_list(l):
        #     res = []
        #     for t in l:
        #         res += t
        #     return res

        # def bachify(data, bs, bptt):
        #     # def bachify_row(row:list, bs, bptt):
        #     res = []
        #     next_word = []
        #     assert len(data[0]) % bs == 0
        #     lb = (len(data[0]) // bs)
        #     for i in range(0, lb, bptt):
        #         j = i + bptt
        #         if j + 1 > lb:
        #             break
        #         batch = [[l[k * lb + i:k * lb + j] for k in range(bs)] for l in data]  # feat -> sent -> seq
        #         batch = list(zip(*batch))
        #         nw_batch = [data[0][k * lb + i + 1:k * lb + j + 1] for k in range(bs)]
        #         res.append(batch)
        #         next_word.append(nw_batch)
        #     return res, next_word

        # if args['bptt'] is not None:
        #     print('Using BPTT of length {}'.format(bptt))
        #     bptt = args['bptt']
        #     bs = max(args['batch_size'] // bptt, 1)
        #     random.shuffle(data)
        #     data = list(zip(*data))
        #     data = [sum_list(l) for l in data]
        #     l_trunc = len(data[0]) - (len(data[0]) % bs)
        #     f_data = [l[:l_trunc] for l in data]
        #     b_data = [l[l_trunc:][::-1] for l in data]
        #     self.f_data, self.f_nw = bachify(f_data, bs, bptt)
        #     self.b_data, self.b_nw = bachify(b_data, bs, bptt)
        #     self.data = self.f_data
        # else:
        #     print('BPTT disabled')
        self.data = self.chunk_batches(data)

        self.num_examples = len(data)

        # chunk into batches

        print("{} batches created for {}.".format(len(self.data), input_srcs))

    def init_vocab(self, data_list):
        assert self.eval == False  # for eval vocab must exist
        data_all = sum(data_list, [])
        charvocab = CharVocab(data_all, self.args['shorthand'])

        # construct wordvocab from multiple files
        wordvocabs = [WordVocab(data, self.args['shorthand'], cutoff=0, lower=True) for data in data_list]
        wordset = list(set(sum([v._id2unit[len(VOCAB_PREFIX):len(VOCAB_PREFIX) + 10000] for v in wordvocabs], [])))
        wordvocab = wordvocabs[0]
        wordvocab._id2unit = VOCAB_PREFIX + wordset
        wordvocab._unit2id = {w: i for i, w in enumerate(wordvocab._id2unit)}
        print('Constructing a joint word vocabulary of size {} ...'.format(len(wordvocab)))

        uposvocab = WordVocab(data_all, self.args['shorthand'], idx=1)
        xposvocab = xpos_vocab_factory(data_all, self.args['shorthand'])
        featsvocab = FeatureVocab(data_all, self.args['shorthand'], idx=3)
        lemmavocab = WordVocab(data_all, self.args['shorthand'], cutoff=self.cutoff, idx=4, lower=True)
        vocab = MultiVocab({'char': charvocab,
                            'word': wordvocab,
                            'upos': uposvocab,
                            'xpos': xposvocab,
                            'feats': featsvocab,
                            'lemma': lemmavocab, })
        return vocab

    def preprocess(self, data, vocab, pretrain_vocab, args):
        """
        returns: list[list[list]] sent -> feat -> seq
        """
        processed = []
        xpos_replacement = [[ROOT_ID] * len(vocab['xpos'])] if isinstance(vocab['xpos'], CompositeVocab) else [ROOT_ID]
        feats_replacement = [[ROOT_ID] * len(vocab['feats'])]
        for sent in data:
            processed_sent = [[ROOT_ID] + vocab['word'].map([w[0] for w in sent])]
            processed_sent += [[[ROOT_ID]] + [vocab['char'].map([x for x in w[0]]) for w in sent]]
            processed_sent += [[ROOT_ID] + vocab['upos'].map([w[1] for w in sent])]
            processed_sent += [xpos_replacement + vocab['xpos'].map([w[2] for w in sent])]
            processed_sent += [feats_replacement + vocab['feats'].map([w[3] for w in sent])]
            processed_sent += [[ROOT_ID] + pretrain_vocab.map([w[0] for w in sent])]
            processed_sent += [[ROOT_ID] + vocab['lemma'].map([w[4] for w in sent])]
            # processed_sent += [[to_int(w[5], ignore_error=self.eval) for w in sent]]
            # processed_sent += [vocab['deprel'].map([w[6] for w in sent])]
            processed.append(processed_sent)

        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == 7

        # sort sentences by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        # sort words by lens for easy char-RNN operations
        batch_words = [w for sent in batch[1] for w in sent]
        word_lens = [len(x) for x in batch_words]
        batch_words, word_orig_idx = sort_all([batch_words], word_lens)
        batch_words = batch_words[0]
        word_lens = [len(x) for x in batch_words]

        # convert to tensors
        words = batch[0]
        words = get_long_tensor(words, batch_size)
        words_mask = torch.eq(words, PAD_ID)
        wordchars = get_long_tensor(batch_words, len(word_lens))
        wordchars_mask = torch.eq(wordchars, PAD_ID)

        upos = get_long_tensor(batch[2], batch_size)
        xpos = get_long_tensor(batch[3], batch_size)
        ufeats = get_long_tensor(batch[4], batch_size)
        pretrained = get_long_tensor(batch[5], batch_size)
        sentlens = [len(x) for x in batch[0]]
        lemma = get_long_tensor(batch[6], batch_size)

        next_word = words.clone()
        next_word[:, :-1] = words[:, 1:]
        next_word[:, -1] = PAD_ID

        prev_word = words.clone()
        prev_word[:, 1:] = words[:, :-1]
        prev_word[:, 0] = PAD_ID

        # if self.args['bptt'] is not None:
        #     next_word = torch.tensor(self.f_nw[key])
        #     prev_word = torch.tensor(self.b_nw[key])

        #     b_batch = self.b_data[key]
        #     b_batch_size = len(b_batch)
        #     b_batch = list(zip(*b_batch))
        #     assert len(b_batch) == 7

        #     # sort sentences by lens for easy RNN operations
        #     b_lens = [len(x) for x in b_batch[0]]
        #     b_batch, orig_idx = sort_all(b_batch, b_lens)

        #     # sort words by lens for easy char-RNN operations
        #     b_batch_words = [w for sent in b_batch[1] for w in sent]
        #     b_word_lens = [len(x) for x in b_batch_words]
        #     b_batch_words, b_word_orig_idx = sort_all([b_batch_words], b_word_lens)
        #     b_batch_words = b_batch_words[0]
        #     b_word_lens = [len(x) for x in b_batch_words]

        #     # convert to tensors
        #     b_words = b_batch[0]
        #     b_words = get_long_tensor(b_words, b_batch_size)
        #     b_words_mask = torch.eq(b_words, PAD_ID)
        #     b_wordchars = get_long_tensor(b_batch_words, len(word_lens))
        #     b_wordchars_mask = torch.eq(wordchars, PAD_ID)

        #     b_upos = get_long_tensor(b_batch[2], b_batch_size)
        #     b_xpos = get_long_tensor(b_batch[3], b_batch_size)
        #     b_ufeats = get_long_tensor(b_batch[4], b_batch_size)
        #     b_pretrained = get_long_tensor(b_batch[5], b_batch_size)
        #     b_sentlens = [len(x) for x in b_batch[0]]
        #     b_lemma = get_long_tensor(b_batch[6], b_batch_size)
        #     return words, words_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained,\
        #         lemma, next_word, None, orig_idx, word_orig_idx, sentlens, word_lens,\
        #         (b_words, b_words_mask, b_wordchars, b_wordchars_mask, b_upos, b_xpos, b_ufeats, b_pretrained,
        #          b_lemma, prev_word, None, b_orig_idx, b_word_orig_idx, b_sentlens, b_word_lens)

        return words, words_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, \
            next_word, prev_word, orig_idx, word_orig_idx, sentlens, word_lens

    def load_file(self, filename, evaluation=False):
        """
        filename: str
        evaluation: (deprecated)

        returns List[List[str]] of shape N x 5
        """
        data = []
        with open(filename, 'r', encoding='utf-8') as fin:
            sent = []
            for line in fin:
                if line.rstrip() == '':
                    data.append(sent)
                    sent = []
                elif len(line.rstrip().split('\t')) != 2:
                    print('Error occur in {}, skipping line'.format(filename))
                else:
                    word, postag = line.rstrip().split('\t')
                    sent.append([word, postag, postag, '_', '_'])
        return data

    def load_doc(self, doc):
        data = doc.conll_file.get(['word', 'upos', 'xpos', 'feats', 'lemma', 'head', 'deprel'], as_sentences=True)
        return doc.conll_file, data

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def reshuffle(self):
        data = [y for x in self.data for y in x]
        self.data = self.chunk_batches(data)
        random.shuffle(self.data)

    def chunk_batches(self, data):
        res = []

        if not self.eval:
            # sort sentences (roughly) by length for better memory utilization
            data = sorted(data, key=lambda x: len(x[0]) + random.randint(0, 3), reverse=random.random() > .5)

        current = []
        currentlen = 0
        for x in data:
            if len(x[0]) + currentlen > self.batch_size:
                res.append(current)
                current = []
                currentlen = 0
            current.append(x)
            currentlen += len(x[0])

        if currentlen > 0:
            res.append(current)

        return res


def to_int(string, ignore_error=False):
    try:
        res = int(string)
    except ValueError as err:
        if ignore_error:
            return 0
        else:
            raise err
    return res
