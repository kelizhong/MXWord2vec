# coding=utf-8
"""collect the tokenized sentence from worker"""
import logbook as logging
import zmq
from utils.appmetric_util import AppMetric
from utils.retry_util import retry


class WordGenerator(object):
    """collect the tokenized sentence from worker
    Parameters
    ----------
        ip : str
            The ip address string without the port to pass to ``Socket.bind()``.
        port:
            The port to receive the tokenized sentence from worker
        tries: int
            Number of times to retry, set to 0 to disable retry
    """

    def __init__(self, ip, port, tries=20, name="word_generator", metric_interval=10):
        self.ip = ip
        self.port = port
        self.tries = tries
        self.name = name
        self.metric_interval = metric_interval

    @retry(lambda x: x.tries, exception=zmq.ZMQError,
           name="word_generator", report=logging.error)
    def _on_recv(self, receiver):
        words = receiver.recv_pyobj(zmq.NOBLOCK)
        return words

    def generate(self):
        """Generator that receive the tokenized sentence from worker and produce the words"""
        context = zmq.Context()
        receiver = context.socket(zmq.PULL)
        receiver.bind("tcp://{}:{}".format(self.ip, self.port))
        metric = AppMetric(name=self.name, interval=self.metric_interval)
        while True:
            try:
                words = self._on_recv(receiver)
            except zmq.ZMQError as e:
                logging.error(e)
                break
            for word in words:
                if len(word):
                    yield word
            metric.notify(1)
