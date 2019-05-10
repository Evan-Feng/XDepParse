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
        print('len 1 = {}'.format(len(data_list[0])))
        print('len 2 = {}'.format(len(data_list[1])))

        # handle vocab
        if vocab is None:
            self.vocab = self.init_vocab(data_list)
        else:
            self.vocab = vocab
        self.pretrain_vocab = pretrain.vocab

        # filter and sample data
        if args.get('sample_train', 1.0) < 1.0 and not self.eval:
            raise NotImplementedError()
            # keep = int(args['sample_train'] * len(data))
            # data = random.sample(data, keep)
            # print("Subsample training set with rate {:g}".format(args['sample_train']))

        data = data_list[0] * 6 + data_list[1] + sum(data_list[2:], [])
        data = self.preprocess(data, self.vocab, self.pretrain_vocab, args)
        # shuffle for training
        if self.shuffled:
            print('shuffling data')
            random.shuffle(data)
        self.num_examples = len(data)

        # chunk into batches
        self.data = self.chunk_batches(data)
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
        print('Joint word vocabulary of size {}'.format(len(wordvocab)))

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

        return words, words_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, next_word, prev_word, orig_idx, word_orig_idx, sentlens, word_lens

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
