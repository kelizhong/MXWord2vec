# coding=utf-8
import random
import mxnet as mx
from utils.pickle_util import load_pickle_object
from exception.resource_exception import ResourceNotFoundException


class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]


class Word2vecDataIter(mx.io.DataIter):
    def __init__(self, vocabulary_path, data_index_path, negative_data_path, batch_size, num_label, data_name='data',
                 label_name='label',
                 label_weight_name='label_weight'):
        super(Word2vecDataIter, self).__init__()
        self.batch_size = batch_size
        vocab, self.data, self.negative = self._load_w2v_data(vocabulary_path, data_index_path, negative_data_path)
        self.vocab_size = len(vocab)
        self.num_label = num_label
        self.data_name = data_name
        self.label_name = label_name
        self.label_weight_name = label_weight_name
        self.provide_data = [(data_name, (batch_size, num_label - 1))]
        self.provide_label = [(label_name, (self.batch_size, num_label)),
                              (label_weight_name, (self.batch_size, num_label))]

    def _load_w2v_data(self, vocabulary_path, data_index_path, negative_data_path):
        vocab = load_pickle_object(vocabulary_path)
        data_index = load_pickle_object(data_index_path)
        negative_data = load_pickle_object(negative_data_path)

        if vocab is None:
            raise ResourceNotFoundException("failed to load vocab")
        if data_index is None:
            raise ResourceNotFoundException("failed to load the data index")
        if negative_data is None:
            raise ResourceNotFoundException("failed to load the negative data")

        return vocab, data_index, negative_data

    def sample_ne(self):
        return self.negative[random.randint(0, len(self.negative) - 1)]

    @property
    def data_names(self):
        return [self.data_name]

    @property
    def label_names(self):
        return [self.label_name] + [self.label_weight_name]

    def __iter__(self):
        batch_data = []
        batch_label = []
        batch_label_weight = []
        start = random.randint(0, self.num_label - 1)
        for i in range(start, len(self.data) - self.num_label - start, self.num_label):
            context = self.data[i: i + self.num_label / 2] \
                      + self.data[i + 1 + self.num_label / 2: i + self.num_label]
            target_word = self.data[i + self.num_label / 2]
            if target_word in [0, 1, 2]:
                continue
            target = [target_word] \
                     + [self.sample_ne() for _ in range(self.num_label - 1)]
            target_weight = [1.0] + [0.0 for _ in range(self.num_label - 1)]
            batch_data.append(context)
            batch_label.append(target)
            batch_label_weight.append(target_weight)
            print batch_data
            if len(batch_data) == self.batch_size:
                data_all = [mx.nd.array(batch_data)]
                label_all = [mx.nd.array(batch_label), mx.nd.array(batch_label_weight)]
                batch_data = []
                batch_label = []
                batch_label_weight = []
                yield SimpleBatch(self.data_names, data_all, self.label_names, label_all)

    def reset(self):
        pass
