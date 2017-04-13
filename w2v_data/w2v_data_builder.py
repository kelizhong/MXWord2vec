# coding=utf-8
"""Generate w2v_data"""
import logbook as logging
from collections import Counter
import gevent.monkey
from utils.pickle_util import save_obj_pickle
from w2v_data.ventilator import VentilatorProcess
from w2v_data.tokenizer_worker import TokenizerWorkerProcess
from w2v_data.word_generator import WordGenerator
from w2v_data.data_index_generator import DataIndexGenerator
from utils.data_util import sentence_gen
from common.constant import unk_word, bos_word, eos_word
import math
import gevent
import gevent.monkey
import time

gevent.monkey.patch_socket()


class Word2vecDataBuilder(object):
    """
    Create w2v_data file (if it does not exist yet) from data file.
    Data file should have one sentence per line.
    Each sentence will be tokenized.
    Vocabulary contains the most-frequent tokens up to top_words.
    We write it to vocab_file in pickle format.
    Parameters
    ----------
        corpus_files: list
            corpus files list that will be used to create w2v_data
        vocab_save_path: str
            vocab file name where the w2v_data will be created
        sentence_gen: generator
            generator which produce the sentence in corpus data
        top_words: int
            limit on the size of the created w2v_data
        workers_num: int
            numbers of workers to parse the sentence
        ip: str
            the ip address string without the port to pass to ``Socket.bind()``
        ventilator_port: int
            port for ventilator process socket
        collector_port: int
            port for collector process socket
        overwrite: bool
            whether to overwrite the existed w2v_data
    """

    def __init__(self, corpus_files, vocab_save_path, data_index_save_path, negative_data_save_path,
                 sentence_gen=sentence_gen, workers_num=1, top_words=100000,
                 ip='127.0.0.1', ventilator_port='5555', collector_port='5556',
                 overwrite=True):
        self.corpus_files = corpus_files
        self.vocab_save_path = vocab_save_path
        self.data_index_save_path = data_index_save_path
        self.negative_data_save_path = negative_data_save_path
        self.sentence_gen = sentence_gen
        self.workers_num = workers_num
        self.top_words = top_words
        self.ip = ip
        self.ventilator_port = ventilator_port
        self.collector_port = collector_port
        self.overwrite = overwrite

    def _start_data_stream_process(self, process_prefix="WorkerProcess"):
        process_pool = []
        v = VentilatorProcess(self.corpus_files, self.ip, self.ventilator_port, sentence_gen=self.sentence_gen)
        v.start()
        process_pool.append(v)
        for i in xrange(self.workers_num):
            w = TokenizerWorkerProcess(self.ip, self.ventilator_port, self.collector_port, name='{}_{}'.format(process_prefix, i))
            w.start()
            process_pool.append(w)
        return process_pool

    def build_and_save_vocabulary(self):
        process_pool = self._start_data_stream_process(process_prefix="VocabularyProcess")
        logging.info("Begin build vocabulary")
        w = WordGenerator(self.ip, self.collector_port)
        vocabulary_counter = Counter(w.generate())
        self._terminate_process(process_pool)
        logging.info("Finish counting. {} unique words, a total of {} words in all files."
                     , len(vocabulary_counter), sum(vocabulary_counter.values()))
        freq = [[unk_word, 0], [bos_word, 0], [eos_word, 0]]
        if self.top_words <= len(freq):
            raise ValueError("vocabulary_size must be larger than {}".format(len(freq)))

        vocabulary_counter = vocabulary_counter.most_common(self.top_words - len(freq))
        freq.extend(vocabulary_counter)
        vocabulary = dict()
        for word, _ in freq:
            vocabulary[word] = len(vocabulary)
        save_obj_pickle(vocabulary, self.vocab_save_path, self.overwrite)
        return vocabulary, freq

    def build_and_save_data_index(self, vocabulary):
        logging.info("Begin build data index")
        data_index = list()
        index_generator = DataIndexGenerator(self.ip, self.collector_port, vocabulary)
        for i, index in enumerate(index_generator.generate()):
            data_index += index
            if i % 10000 == 0:
                logging.info("Have processed {} data index".format(i))
        save_obj_pickle(data_index, self.data_index_save_path, self.overwrite)
        logging.info("Finish build data index")

    def build_and_save_negative_data(self, vocabulary_frequency):
        logging.info("Begin build negative data")
        negative_data = list()
        for i, v in enumerate(vocabulary_frequency):
            count = v[1]
            if i == 0:
                continue
            count = int(math.pow(count * 1.0, 0.75))
            negative_data += [i for _ in range(count)]
            if i % 10000 == 0:
                logging.info("Have processed {} negative data".format(i))
        save_obj_pickle(negative_data, self.negative_data_save_path, self.overwrite)
        logging.info("Finish build negative data")

    def build_word2vec_dataset_and_vocabulary(self):

        vocabulary, vocabulary_frequency = self.build_and_save_vocabulary()

        process_pool = self._start_data_stream_process(process_prefix="DataProcess")

        gevent.joinall(
            [gevent.spawn(self.build_and_save_data_index, vocabulary),
             gevent.spawn(self.build_and_save_negative_data, vocabulary_frequency)])
        self._terminate_process(process_pool)

    def _terminate_process(self, pool):
        for p in pool:
            p.terminate()
            logging.info('terminated process {}', p.name)
