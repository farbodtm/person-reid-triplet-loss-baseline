from .Dataset import Dataset
from ..utils.dataset_utils import parse_im_name

import os.path as osp
from PIL import Image
import numpy as np
import random
from collections import defaultdict


class TrainSet(Dataset):
  """Training set for triplet loss.
  Args:
    ids2labels: a dict mapping ids to labels
  """

  def __init__(
      self,
      im_dir=None,
      im_names=None,
      ids2labels=None,
      num_classes=None,
      num_shots=None,
      batch_size=None,
      **kwargs):

    # The im dir of all images
    self.im_dir = im_dir
    self.im_names = im_names
    self.ids2labels = ids2labels
    self.num_classes = num_classes
    self.num_shots = num_shots

    im_ids = [parse_im_name(name, 'id') for name in im_names]
    self.ids_to_im_inds = defaultdict(list)
    for ind, id in enumerate(im_ids):
      self.ids_to_im_inds[id].append(ind)
    self.ids = self.ids_to_im_inds.keys()
    print(self.ids)

    super(TrainSet, self).__init__(
      dataset_size=len(self.ids),
      batch_size=batch_size,
      **kwargs)

  def get_sample(self, i=0):
    """Here one sample means several images (and labels etc) of one id.
    Returns:
      ims: a list of images
    """
    rid = random.sample(self.ids, 1)[0]
    inds = self.ids_to_im_inds[rid]
    if len(inds) < self.num_shots:
      inds = np.random.choice(inds, self.num_shots, replace=True)
    else:
      inds = np.random.choice(inds, self.num_shots, replace=False)
    im_names = [self.im_names[ind] for ind in inds]
    ims = [np.asarray(Image.open(osp.join(self.im_dir, name)))
           for name in im_names]
    ims, mirrored = zip(*[self.pre_process_im(im) for im in ims])
    labels = [self.ids2labels[rid] for _ in range(self.num_shots)]
    ilabels = [i for _ in range(self.num_shots)]
    return ims, im_names, labels, ilabels, mirrored

  def mini_dataset(self):
    samples = []
    for _ in range(self.num_classes):
      samples.append(self.get_sample(_))

    return samples 

  def mini_batches(self, samples, num_batches, replacement=False):
    """
    Generate mini-batches from some data.

    Returns:
      An iterable of sequences of (input, label) pairs,
        where each sequence is a mini-batch.
    """
    im_list, im_names, labels, ilabels, mirrored = zip(*samples)
    ims = np.stack(np.concatenate(im_list))
    im_names = np.concatenate(im_names)
    labels = np.concatenate(labels)
    ilabels = np.concatenate(ilabels)
    mirrored = np.concatenate(mirrored)
    samples = zip(ims, im_names, labels, ilabels, mirrored)

    samples = list(samples)
    if replacement:
      for _ in range(num_batches):
        yield random.sample(samples, batch_size)
      return
    cur_batch = []
    batch_count = 0
    while True:
      random.shuffle(samples)
      for sample in samples:
        cur_batch.append(sample)
        if len(cur_batch) < self.batch_size:
          continue
        yield cur_batch
        cur_batch = []
        batch_count += 1
        if batch_count == num_batches:
          return

  def next_batch(self):
    """Next batch of images and labels.
    Returns:
      ims: numpy array with shape [N, H, W, C] or [N, C, H, W], N >= 1
      img_names: a numpy array of image names, len(img_names) >= 1
      labels: a numpy array of image labels, len(labels) >= 1
      mirrored: a numpy array of booleans, whether the images are mirrored
      self.epoch_done: whether the epoch is over
    """
    return
