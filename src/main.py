import argparse
import datetime
import os
import random
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from datasets import create_dataset, TCCDataLoader, load_videos
from losses import temporal_cycle_consistency_loss
from utils import get_logger, get_model, get_optimizer, save_checkpoint


def main(args):
    random.seed(args.random_state)
    os.environ['PYTHONHASHSEED'] = str(args.random_state)
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)

    # logs
    expid = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    args.logdir = os.path.join(args.logdir, expid)
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    os.chmod(args.logdir, 0o777)
    logger = get_logger(os.path.join(args.logdir, 'main.log'))
    logger.info(args)
    writer = SummaryWriter(args.logdir)

    # data
    t0 = time.time()
    videos, video_seq_lens = load_videos(args.data_path, args.input_size)
    assert len(videos) == len(video_seq_lens) == 70  # TODO: only for the pouring dataset
    print('Loaded {} videos in {:.2f}s'.format(len(videos), time.time() - t0))
    print(videos.shape)
    print(max(video_seq_lens))
    dataset = create_dataset(videos, video_seq_lens, args.num_frames, args.num_context_steps, args.context_stride)
    dataloader = TCCDataLoader(dataset, args.batch_size)

    # model
    model = get_model(args).to(args.device)
    optimizer = get_optimizer(model, args)

    # train
    iter_i = 0
    for frames, steps, seq_lens in dataloader:
        iter_i += 1

        frames = frames.to(args.device)
        steps = steps.to(args.device)
        seq_lens = seq_lens.to(args.device)

        embeddings = model(frames)
        loss = temporal_cycle_consistency_loss(
            embeddings, steps, seq_lens,
            num_frames=args.num_frames, batch_size=args.batch_size,
            temperature=args.temperature, variance_lambda=args.variance_lambda,
            normalize_indices=args.normalize_indices,
            args=args,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (iter_i % args.checkpoint_freq) == 0:
            state_dict = {
                'iter_i': iter_i,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss/train': loss.item(),
            }
            filename = f'checkpoint_{iter_i:05}_loss-{loss.item():.4f}.pt'
            save_checkpoint(state_dict, args.logdir, filename)
        writer.add_scalar('Loss/Train', loss.item(), iter_i)
        if (iter_i % 10) == 0:
            logger.info('[{}] Iter {} Loss {}'.format(expid, iter_i, loss.item()))
        if iter_i > args.num_training_iters:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--num_frames', type=int, default=32)
    parser.add_argument('--num_context_steps', type=int, default=2)
    parser.add_argument('--context_stride', type=int, default=15)
    parser.add_argument('--input_size', nargs='+', type=int, default=(168, 168))
    # model
    parser.add_argument('--base_model_name', type=str, default='resnet50')
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--embedding_size', type=int, default=128)
    # loss
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--variance_lambda', type=float, default=1e-3)
    parser.add_argument('--normalize_indices', action='store_true', default=False)
    # optim
    parser.add_argument('--optim_name', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    # training
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--num_training_iters', type=int, default=10000)
    parser.add_argument('--checkpoint_freq', type=int, default=500)
    # misc
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--random_state', type=int, default=42)

    args, unknown_args = parser.parse_known_args()

    args.data_path = '../data/pouring/'
    args.logdir = '../logs/'
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.normalize_indices = True
    args.batch_size = 8

    main(args)
