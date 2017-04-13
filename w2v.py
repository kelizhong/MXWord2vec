# coding=utf-8
"""
Runner for Query2Vec

"""

import os
import sys
import argparse
import logbook as logging
import signal
from argparser.customArgType import FileType
from utils.data_util import sentence_gen
from utils.log_util import Logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train Seq2seq query2vec for query2vec')
    parser.add_argument('--log-file-name', default=os.path.join(os.getcwd(), 'data/logs', 'q2v.log'),
                        type=FileType, help='Log directory (default: __DEFAULT__).')
    parser.add_argument('--metric-interval', default=6, type=int,
                        help='metric reporting frequency is set by seconds param')
    subparsers = parser.add_subparsers(help='train vocabulary')

    w2v_vocab_parser = subparsers.add_parser("w2v_vocab")
    w2v_vocab_parser.set_defaults(action='w2v_vocab')

    # vocabulary parameter

    w2v_vocab_parser.add_argument('--overwrite', action='store_true', help='overwrite earlier created files, also forces the\
                        program not to reuse count files')
    w2v_vocab_parser.add_argument('--negative-data-save-path',
                                  type=FileType,
                                  default=os.path.join(os.path.dirname(__file__), 'data', 'negative_data.pkl'),
                                  help='the file with the words which are the most command words in the corpus')
    w2v_vocab_parser.add_argument('--vocab-save-file',
                                  type=FileType,
                                  default=os.path.join(os.path.dirname(__file__), 'data', 'vocab.pkl'),
                                  help='the file with the words which are the most command words in the corpus')
    w2v_vocab_parser.add_argument('--data-index-save-path',
                                  type=FileType,
                                  default=os.path.join(os.path.dirname(__file__), 'data', 'data_index.pkl'),
                                  help='the file with the words which are the most command words in the corpus')
    w2v_vocab_parser.add_argument('-w', '--workers-num',
                                  type=int,
                                  default=1,
                                  help='the file with the words which are the most command words in the corpus')
    w2v_vocab_parser.add_argument('--top-words', default=1000000, type=int,
                                  help='the max words num for training')
    w2v_vocab_parser.add_argument('files', nargs='+',
                                  help='the corpus input files')

    return parser.parse_args()


def signal_handler(signal, frame):
    logging.info('Stop!!!')
    sys.exit(0)


def setup_logger():
    log = Logger()
    log.set_stream_handler()
    log.set_time_rotating_file_handler(args.log_file_name)


if __name__ == "__main__":
    args = parse_args()
    setup_logger()
    signal.signal(signal.SIGINT, signal_handler)
    if args.action == 'w2v_vocab':
        from w2v_data.w2v_data_builder import Word2vecDataBuilder

        w2v_data = Word2vecDataBuilder(args.files, args.vocab_save_file, args.data_index_save_path,
                                       args.negative_data_save_path, workers_num=args.workers_num,
                                       sentence_gen=sentence_gen,
                                       overwrite=args.overwrite, top_words=args.top_words)

        w2v_data.build_word2vec_dataset_and_vocabulary()
