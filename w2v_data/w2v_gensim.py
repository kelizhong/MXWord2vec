# coding=utf-8
"""Generate w2v_data"""
import logbook as logging
from w2v_data.ventilator import VentilatorProcess
from w2v_data.tokenizer_worker import TokenizerWorkerProcess
from w2v_data.gensim_word_generator import WordGenerator
from utils.data_util import sentence_gen
import gensim
import multiprocessing


class GensimWord2vec(object):
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

    def build_word2vec_dataset_and_vocabulary(self):

        process_pool = self._start_data_stream_process(process_prefix="VocabularyProcess")
        logging.info("Begin build vocabulary")
        w = WordGenerator(self.ip, self.collector_port)
        model = gensim.models.Word2Vec(w, size=400, window=5, min_count=20,
                                       workers=multiprocessing.cpu_count()-4)
        model.save('./data/w2v_gensim')
        model.save_word2vec_format('./data/w2v_gensim_text', binary=False)
        self._terminate_process(process_pool)

    def _terminate_process(self, pool):
        for p in pool:
            p.terminate()
            logging.info('terminated process {}', p.name)
