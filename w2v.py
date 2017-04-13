# coding=utf-8
"""
Runner for Query2Vec

"""

import os
import sys
import argparse
import logbook as logging
import signal
from argparser.customArgType import FileType, DirectoryType
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
    w2v_trainer_parser = subparsers.add_parser("train_w2v")
    w2v_trainer_parser.set_defaults(action='train_w2v')

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

    # word2vec parameter

    # mxnet parameter
    w2v_trainer_parser.add_argument('-dm', '--device-mode', choices=['cpu', 'gpu', 'gpu_auto'],
                                    help='define define mode, (default: %(default)s)',
                                    default='cpu')
    w2v_trainer_parser.add_argument('-d', '--devices', type=str, default='0',
                                    help='the devices will be used, e.g "0,1,2,3"')

    w2v_trainer_parser.add_argument('-lf', '--log-freq', default=1000, type=int,
                                    help='the frequency to printout the training verbose information')

    w2v_trainer_parser.add_argument('-scf', '--save-checkpoint-freq', default=1, type=int,
                                    help='the frequency to save checkpoint')

    w2v_trainer_parser.add_argument('-kv', '--kv-store', dest='kv_store', help='the kv-store type',
                                    default='local', type=str)
    w2v_trainer_parser.add_argument('-mi', '--monitor-interval', default=0, type=int,
                                    help='log network parameters every N iters if larger than 0')
    w2v_trainer_parser.add_argument('-eval', '--enable-evaluation', action='store_true', help='enable evaluation')

    w2v_trainer_parser.add_argument('-wll', '--work-load-ist', dest='work_load_list',
                                    help='work load for different devices',
                                    default=None, type=list)
    w2v_trainer_parser.add_argument('--hosts-num', dest='hosts_num', help='the number of the hosts',
                                    default=1, type=int)
    w2v_trainer_parser.add_argument('--workers-num', dest='workers_num', help='the number of the workers',
                                    default=1, type=int)
    w2v_trainer_parser.add_argument('-db', '--disp-batches', dest='disp_batches',
                                    help='show progress for every n batches',
                                    default=1, type=int)
    w2v_trainer_parser.add_argument('-le', '--load-epoch', dest='load_epoch', help='epoch of pretrained model',
                                    type=int, default=-1)
    w2v_trainer_parser.add_argument('-r', '--rank', dest='rank', help='epoch of pretrained model',
                                    type=int, default=0)
    w2v_trainer_parser.add_argument('-mp', '--model-prefix', default='word2vec',
                                    type=str,
                                    help='the experiment name, this is also the prefix for the parameters file')
    w2v_trainer_parser.add_argument('-pd', '--model-path',
                                    default=os.path.join(os.getcwd(), 'data', 'word2vec', 'model'),
                                    type=DirectoryType,
                                    help='the directory to store the parameters of the training')

    # optimizer parameter
    w2v_trainer_parser.add_argument('-opt', '--optimizer', type=str, default='AdaGrad',
                                    help='the optimizer type')
    w2v_trainer_parser.add_argument('-cg', '--clip-gradient', type=float, default=5.0,
                                    help='clip gradient in range [-clip_gradient, clip_gradient]')
    w2v_trainer_parser.add_argument('--wd', type=float, default=0.00001,
                                    help='weight decay for sgd')
    w2v_trainer_parser.add_argument('--mom', dest='momentum', type=float, default=0.9,
                                    help='momentum for sgd')
    w2v_trainer_parser.add_argument('--lr', dest='learning_rate', type=float, default=0.01,
                                    help='initial learning rate')

    # model parameter
    w2v_trainer_parser.add_argument('-bs', '--batch-size', default=128, type=int,
                                    help='batch size for each databatch')
    w2v_trainer_parser.add_argument('-es', '--embed-size', default=128, type=int,
                                    help='embedding size ')
    w2v_trainer_parser.add_argument('-ws', '--window-size', default=2, type=int,
                                    help='window size ')

    # word2vec data parameter
    w2v_trainer_parser.add_argument('vocabulary_path', type=str,
                                    help='the file name of the corpus')
    w2v_trainer_parser.add_argument('data_index_path', type=str,
                                    help='the file name of the corpus')
    w2v_trainer_parser.add_argument('negative_data_path', type=str,
                                    help='the file name of the corpus')
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
    elif args.action == 'train_w2v':
        from word2vec.word2vec_trainer import Word2vecTrainer, mxnet_parameter, optimizer_parameter, model_parameter

        mxnet_para = mxnet_parameter(kv_store=args.kv_store, hosts_num=args.hosts_num, workers_num=args.workers_num,
                                     device_mode=args.device_mode, devices=args.devices,
                                     disp_batches=args.disp_batches, monitor_interval=args.monitor_interval,
                                     save_checkpoint_freq=args.save_checkpoint_freq,
                                     model_path_prefix=os.path.join(args.model_path, args.model_prefix),
                                     enable_evaluation=args.enable_evaluation,
                                     load_epoch=args.load_epoch)

        optimizer_para = optimizer_parameter(optimizer=args.optimizer, learning_rate=args.learning_rate, wd=args.wd,
                                             momentum=args.momentum)

        model_para = model_parameter(embed_size=args.embed_size, batch_size=args.batch_size,
                                     window_size=args.window_size)
        Word2vecTrainer(vocabulary_path=args.vocabulary_path, data_index_path=args.data_index_path, negative_data_path=args.negative_data_path,
                        mxnet_para=mxnet_para,
                        optimizer_para=optimizer_para,
                        model_para=model_para).train()
