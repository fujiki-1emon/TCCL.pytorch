import os

import torch
from torch import optim

from models import Embedder


def get_model(args):
    model = Embedder(args.embedding_size, args.num_context_steps, args=args)
    return model


def get_optimizer(model, args):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return optimizer


def save_checkpoint(state, logdir, filename='checkpoint.pt'):
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    path = os.path.join(logdir, filename)
    torch.save(state, path)


def get_logger(log_file):
    from logging import getLogger, FileHandler, StreamHandler
    from logging import Formatter, DEBUG, ERROR, INFO  # noqa
    fh = FileHandler(log_file)
    fh.setLevel(INFO)
    sh = StreamHandler()
    sh.setLevel(INFO)

    for handler in [fh, sh]:
        formatter = Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
    logger = getLogger('log')
    logger.setLevel(DEBUG)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger
