# coding=utf-8
"""Worker process which receiver the sentence from ventilitor process and tokenize it"""
import logbook as logging
from multiprocessing import Process
import zmq
from zmq.decorators import socket
from utils.data_util import tokenize
from utils.appmetric_util import AppMetric
from utils.retry_util import retry


class WorkerProcess(Process):
    """Worker process which receiver the sentence from ventilitor process and tokenize it
    Parameters
    ----------
        ip : str
            The ip address string without the port to pass to ``Socket.bind()``.
        frontend_port: int
            Port for the incoming traffic
        backend_port: int
            Port for the outbound traffic
        tries: int
            Number of times to retry, set to 0 to disable retry
        name: str
            Process name
    """

    def __init__(self, ip, frontend_port, backend_port, tries=10, name='WorkerProcess', metric_interval=10):
        Process.__init__(self)
        self.ip = ip
        self.frontend_port = frontend_port
        self.backend_port = backend_port
        self.tries = tries
        self.name = name
        self.metric_interval = metric_interval

    @retry(lambda x: x.tries, exception=zmq.ZMQError, name='worker_parser', report=logging.error)
    def _on_recv(self, receiver):
        sentence = receiver.recv_string(zmq.NOBLOCK)
        return sentence

    @socket(zmq.PULL)
    @socket(zmq.PUSH)
    def run(self, receiver, sender):
        receiver.connect("tcp://{}:{}".format(self.ip, self.frontend_port))
        sender.connect("tcp://{}:{}".format(self.ip, self.backend_port))
        metric = AppMetric(name=self.name, interval=self.metric_interval)
        while True:
            try:
                sentence = self._on_recv(receiver)
            except zmq.ZMQError as e:
                logging.error(e)
                break
            tokens = tokenize(sentence)
            sender.send_pyobj(tokens)
            metric.notify(1)
