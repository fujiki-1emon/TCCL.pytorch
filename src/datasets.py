import glob
import os

import cv2
import numpy as np
import tensorflow as tf
import torch


class TCCDataLoader():
    def __init__(self, ds, batch_size, args=None):
        self.ds = ds.batch(batch_size)
        self.ds = self.ds.prefetch(1)
        self.ds = self.ds.as_numpy_iterator()

    def __len__(self):
        return 70  # TODO: only for pouring dataset

    def __iter__(self):
        return self

    def __next__(self):
        batch = next(self.ds)
        frames, steps, seq_lens = batch['frames'], batch['steps'], batch['seq_lens']
        frames = torch.from_numpy(frames).permute(0, 1, 4, 2, 3)  # (bs, ts, c, h, w)
        steps = torch.from_numpy(steps)
        seq_lens = torch.from_numpy(seq_lens)
        return frames, steps, seq_lens


def create_dataset(videos, seq_lens, num_steps, num_context_steps, context_stride):
    ds = tf.data.Dataset.from_tensor_slices((videos, seq_lens))
    ds = ds.repeat()
    ds = ds.shuffle(len(videos))

    def sample_and_preprocess(video, seq_len):
        steps = tf.sort(tf.random.shuffle(tf.range(seq_len))[:num_steps])

        def get_context_steps(step):
            return tf.clip_by_value(
                tf.range(step - (num_context_steps - 1) * context_stride, step + context_stride, context_stride),
                0, seq_len - 1
            )

        steps_with_context = tf.reshape(tf.map_fn(get_context_steps, steps), [-1])
        frames = tf.gather(video, steps_with_context)
        frames = tf.cast(frames, tf.float16)
        frames = (frames / 127.5) - 1.0
        frames = tf.image.resize(frames, (168, 168))
        return {'frames': frames, 'seq_lens': seq_len, 'steps': steps}

    ds = ds.map(sample_and_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return ds


def read_video(video_filename, width=168, height=168):
    cap = cv2.VideoCapture(video_filename)
    frames = []
    if cap.isOpened():
        while True:
            success, frame_bgr = cap.read()
            if not success:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (width, height))
            frames.append(frame_rgb)
    frames = np.asarray(frames)
    return frames


def pad_zeros(frames, max_seq_len):
    npad = ((0, max_seq_len - len(frames)), (0, 0), (0, 0), (0, 0))
    frames = np.pad(frames, pad_width=npad, mode='constant', constant_values=0)
    return frames


def load_videos(path_to_raw_videos, size):
    video_filenames = sorted(glob.glob(os.path.join(path_to_raw_videos, '*.mp4')))

    videos = []
    video_seq_lens = []
    for video_filename in video_filenames:
        frames = read_video(video_filename, width=size[0], height=size[1])
        videos.append(frames)
        video_seq_lens.append(len(frames))
    max_seq_len = max(video_seq_lens)
    videos = np.asarray([pad_zeros(x, max_seq_len) for x in videos])
    return videos, video_seq_lens
