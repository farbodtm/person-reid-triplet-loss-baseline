from __future__ import print_function

import sys

sys.path.insert(0, '.')

import torch
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.parallel import DataParallel

import time
import os.path as osp
from tensorboardX import SummaryWriter
import numpy as np
import argparse

from reptile.dataset import create_dataset
from reptile.model.Model import Model
import reptile.model.meta as meta

from reptile.utils.utils import time_str
from reptile.utils.utils import str2bool
from reptile.utils.utils import tight_float_str as tfs
from reptile.utils.utils import may_set_mode
from reptile.utils.utils import load_state_dict
from reptile.utils.utils import load_ckpt
from reptile.utils.utils import save_ckpt
from reptile.utils.utils import set_devices
from reptile.utils.utils import AverageMeter
from reptile.utils.utils import to_scalar
from reptile.utils.utils import ReDirectSTD
from reptile.utils.utils import set_seed
from reptile.utils.utils import adjust_lr_exp
from reptile.utils.utils import adjust_lr_staircase


class Config(object):
  def __init__(self):

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--sys_device_ids', type=eval, default=(0,))
    parser.add_argument('-r', '--run', type=int, default=1)
    parser.add_argument('--set_seed', type=str2bool, default=False)
    parser.add_argument('--dataset', type=str, default='market1501',
                        choices=['market1501', 'cuhk03', 'duke', 'combined'])
    parser.add_argument('--trainset_part', type=str, default='trainval',
                        choices=['trainval', 'train'])

    parser.add_argument('--resize_h_w', type=eval, default=(256, 128))
    # These several only for training set
    parser.add_argument('--crop_prob', type=float, default=0)
    parser.add_argument('--crop_ratio', type=float, default=1)
    parser.add_argument('--mirror', type=str2bool, default=True)

    parser.add_argument('--log_to_file', type=str2bool, default=True)
    parser.add_argument('--steps_per_log', type=int, default=20)
    parser.add_argument('--steps_per_val', type=int, default=1e3)

    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--num_shots', type=int, default=5)
    parser.add_argument('--traion_shots', type=int, default=15)

    parser.add_argument('--meta_train_iters', type=int, default=60000)
    parser.add_argument('--meta_train_step_size', type=int, default=1)
    parser.add_argument('--meta_train_inner_batch_size', type=int, default=8)
    parser.add_argument('--meta_train_inner_iters', type=int, default=10)
    parser.add_argument('--meta_train_batch_size', type=int, default=5)

    parser.add_argument('--learning-rate', type=float, default=0.00022)

    parser.add_argument('--meta_eval_inner_batch_size', type=int, default=8)
    parser.add_argument('--meta_eval_inner_iters', type=int, default=10)
    parser.add_argument('--meta_eval_batch_size', type=int, default=5)

    parser.add_argument('--normalize_feature', type=str2bool, default=False)

    parser.add_argument('--only_test', type=str2bool, default=False)
    parser.add_argument('--resume', type=str2bool, default=False)
    parser.add_argument('--exp_dir', type=str, default='')
    parser.add_argument('--model_weight_file', type=str, default='')

    args = parser.parse_args()

    # gpu ids
    self.sys_device_ids = args.sys_device_ids

    # If you want to make your results exactly reproducible, you have
    # to fix a random seed.
    if args.set_seed:
      self.seed = 1
    else:
      self.seed = None

    # The experiments can be run for several times and performances be averaged.
    # `run` starts from `1`, not `0`.
    self.run = args.run

    ###########
    # Dataset #
    ###########

    # If you want to make your results exactly reproducible, you have
    # to also set num of threads to 1 during training.
    if self.seed is not None:
      self.prefetch_threads = 1
    else:
      self.prefetch_threads = 2

    self.dataset = args.dataset
    self.trainset_part = args.trainset_part

    # Image Processing

    # Just for training set
    self.crop_prob = args.crop_prob
    self.crop_ratio = args.crop_ratio
    self.resize_h_w = args.resize_h_w

    # Whether to scale by 1/255
    self.scale_im = True

    #self.im_mean = [0.486, 0.459, 0.408]
    #self.im_std = [0.229, 0.224, 0.225]

    self.im_mean = None
    self.im_std = None

    self.train_mirror_type = 'random' if args.mirror else None

    #self.ids_per_batch = args.ids_per_batch
    #self.ims_per_id = args.ims_per_id
    self.num_classes = args.num_classes
    self.train_shots = args.num_shots

    self.train_batch_size=args.meta_train_batch_size

    self.train_shuffle = True

    self.test_batch_size = 32
    self.test_final_batch = True
    self.test_mirror_type = None
    self.test_shuffle = False

    dataset_kwargs = dict(
      name=self.dataset,
      resize_h_w=self.resize_h_w,
      scale=self.scale_im,
      im_mean=self.im_mean,
      im_std=self.im_std,
      batch_dims='NCHW',
      num_prefetch_threads=self.prefetch_threads)

    prng = np.random
    if self.seed is not None:
      prng = np.random.RandomState(self.seed)
    self.train_set_kwargs = dict(
      part=self.trainset_part,
      num_classes=self.num_classes,
      num_shots=self.train_shots,
      batch_size=self.train_batch_size,
      shuffle=self.train_shuffle,
      crop_prob=self.crop_prob,
      crop_ratio=self.crop_ratio,
      mirror_type=self.train_mirror_type,
      prng=prng)
    self.train_set_kwargs.update(dataset_kwargs)
    del self.train_set_kwargs['num_prefetch_threads']

    prng = np.random
    if self.seed is not None:
      prng = np.random.RandomState(self.seed)
    self.val_set_kwargs = dict(
      part='val',
      num_classes=self.num_classes,
      num_shots=self.train_shots,
      eval_batch_size=self.meta_eval_batch_size,
      batch_size=self.test_batch_size,
      final_batch=self.test_final_batch,
      shuffle=self.test_shuffle,
      mirror_type=self.test_mirror_type,
      prng=prng)
    self.val_set_kwargs.update(dataset_kwargs)

    prng = np.random
    if self.seed is not None:
      prng = np.random.RandomState(self.seed)
    self.test_set_kwargs = dict(
      part='test',
      num_classes=self.num_classes,
      num_shots=self.train_shots,
      eval_batch_size=self.meta_eval_batch_size,
      batch_size=self.test_batch_size,
      final_batch=self.test_final_batch,
      shuffle=self.test_shuffle,
      mirror_type=self.test_mirror_type,
      prng=prng)
    self.test_set_kwargs.update(dataset_kwargs)

    ###############
    # ReID Model  #
    ###############

    # Whether to normalize feature to unit length along the Channel dimension,
    # before computing distance
    self.normalize_feature = args.normalize_feature

    #############
    # Training  #
    #############

    self.meta_train_iters = args.meta_train_iters
    self.meta_train_step_size = args.meta_train_step_size
    self.meta_train_inner_batch_size = args.meta_train_inner_batch_size
    self.meta_train_inner_iters = args.meta_train_inner_iters
    self.meta_train_batch_size = args.meta_train_batch_size
    self.learning_rate = args.learning_rate

    self.meta_eval_inner_batch_size = args.meta_eval_inner_batch_size
    self.meta_eval_inner_iters = args.meta_eval_inner_iters
    self.meta_eval_batch_size = args.meta_eval_batch_size

    # How often (in epochs) to test on val set.
    self.stpes_per_val = args.steps_per_val

    # How often (in batches) to log. If only need to log the average
    # information for each epoch, set this to a large value, e.g. 1e10.
    self.steps_per_log = args.steps_per_log

    # Only test and without training.
    self.only_test = args.only_test

    self.resume = args.resume

    #######
    # Log #
    #######

    # If True,
    # 1) stdout and stderr will be redirected to file,
    # 2) training loss etc will be written to tensorboard,
    # 3) checkpoint will be saved
    self.log_to_file = args.log_to_file

    # The root dir of logs.
    if args.exp_dir == '':
      self.exp_dir = osp.join(
        'exp/train',
        '{}'.format(self.dataset),
        #
        'lr_{}_'.format(tfs(self.learning_rate)) +
        'mstep_{}_'.format(tfs(self.meta_train_step)) +
        'total_{}'.format(self.meta_train_iters),
        #
        'run{}'.format(self.run),
      )
    else:
      self.exp_dir = args.exp_dir

    self.stdout_file = osp.join(
      self.exp_dir, 'stdout_{}.txt'.format(time_str()))
    self.stderr_file = osp.join(
      self.exp_dir, 'stderr_{}.txt'.format(time_str()))

    # Saving model weights and optimizer states, for resuming.
    self.ckpt_file = osp.join(self.exp_dir, 'ckpt.pth')
    # Just for loading a pretrained model; no optimizer states is needed.
    self.model_weight_file = args.model_weight_file


class ExtractFeature(object):
  """A function to be called in the val/test set, to extract features.
  Args:
    TVT: A callable to transfer images to specific device.
  """

  def __init__(self, model, TVT):
    self.model = model
    self.TVT = TVT

  def __call__(self, ims):
    old_train_eval_model = self.model.training
    # Set eval mode.
    # Force all BN layers to use global mean and variance, also disable
    # dropout.
    #self.model.eval()
    ims = Variable(self.TVT(torch.from_numpy(ims).float()))
    feat = self.model.embedding(ims)
    feat = feat.data.cpu().numpy()
    # Restore the model to its old train/eval mode.
    #self.model.train(old_train_eval_model)
    return feat


def main():
  cfg = Config()

  # Redirect logs to both console and file.
  if cfg.log_to_file:
    ReDirectSTD(cfg.stdout_file, 'stdout', False)
    ReDirectSTD(cfg.stderr_file, 'stderr', False)

  # Lazily create SummaryWriter
  writer = None

  TVT, TMO = set_devices(cfg.sys_device_ids)

  if cfg.seed is not None:
    set_seed(cfg.seed)

  # Dump the configurations to log.
  import pprint
  print('-' * 60)
  print('cfg.__dict__')
  pprint.pprint(cfg.__dict__)
  print('-' * 60)

  ###########
  # Dataset #
  ###########

  if not cfg.only_test:
    train_set = create_dataset(**cfg.train_set_kwargs)
    val_set = create_dataset(**cfg.val_set_kwargs)

  test_sets = []
  test_set_names = []
  if cfg.dataset == 'combined':
    for name in ['market1501', 'cuhk03', 'duke']:
      cfg.test_set_kwargs['name'] = name
      test_sets.append(create_dataset(**cfg.test_set_kwargs))
      test_set_names.append(name)
  else:
    test_sets.append(create_dataset(**cfg.test_set_kwargs))
    test_set_names.append(cfg.dataset)

  ###########
  # Models  #
  ###########

  model = Model(cfg.num_classes)
  # Model wrapper
  if torch.cuda.is_available():
    model.cuda()
  #model_w = DataParallel(model)

  #############################
  # Criteria and Optimizers   #
  #############################

  #tri_loss = TripletLoss(margin=cfg.margin)

  #optimizer = optim.Adam(model.parameters(),
  #                       lr=cfg.base_lr,
  #                       weight_decay=cfg.weight_decay)

  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(),
      lr=cfg.learning_rate,
      betas=(0, 0.999))

  # Bind them together just to save some codes in the following usage.
  modules_optims = [model, optimizer]

  ################################
  # May Resume Models and Optims #
  ################################

  if cfg.resume:
    resume_ep, scores = load_ckpt(modules_optims, cfg.ckpt_file)

  # May Transfer Models and Optims to Specified Device. Transferring optimizer
  # is to cope with the case when you load the checkpoint to a new device.
  TMO(modules_optims)

  ########
  # Test #
  ########

  def test(load_model_weight=False):
    if load_model_weight:
      if cfg.model_weight_file != '':
        map_location = (lambda storage, loc: storage)
        sd = torch.load(cfg.model_weight_file, map_location=map_location)
        load_state_dict(model, sd)
        print('Loaded model weights from {}'.format(cfg.model_weight_file))
      else:
        load_ckpt(modules_optims, cfg.ckpt_file)

    for test_set, name in zip(test_sets, test_set_names):
      weights_original = meta.eval_tarin(test_set, model, criterion, optimizer, cfg.meta_eval_inner_iters, cfg.eval_iters)
      test_set.set_feat_func(ExtractFeature(model, TVT))
      print('\n=========> Test on dataset: {} <=========\n'.format(name))
      test_set.eval(
        normalize_feat=cfg.normalize_feature,
        verbose=True)
      meta.reset_weights(model, weights_original)

  def validate():
    weights_original = meta.eval_tarin(val_set, model, criterion, optimizer, cfg.meta_eval_inner_iters, cfg.eval_iters)
    if val_set.extract_feat_func is None:
      val_set.set_feat_func(ExtractFeature(model, TVT))
    print('\n=========> Test on validation set <=========\n')
    mAP, cmc_scores, _, _ = val_set.eval(
      normalize_feat=cfg.normalize_feature,
      to_re_rank=False,
      verbose=False)
    print()
    meta.reset_weights(model, weights_original)
    return mAP, cmc_scores[0]

  if cfg.only_test:
    test(load_model_weight=True)
    return

  ############
  # Training #
  ############

  start_it = resume_ep if cfg.resume else 0

  for step in range(start_it, cfg.meta_train_iters):
    loss_meter = AverageMeter()

    step_st = time.time()

    frac_done = float(step) / cfg.meta_train_iters
    current_step_size = cfg.meta_train_step_size * (1. - frac_done)

    meta.meta_train_step(train_set, model, criterion, optimizer, cfg.meta_train_inner_iters, current_step_size, cfg.meta_train_batch_size)
    print('step done')
    exit()

    ############
    # Step Log #
    ############

    if step % cfg.steps_per_log == 0:
      time_log = '\tStep {}, {:.2f}s'.format(
          step+1, steptime.time() - step_st, )

      print(log)

    ##########################
    # Test on Validation Set #
    ##########################

    mAP, Rank1 = 0, 0
    if (step + 1) % cfg.epochs_per_val == 0:
      mAP, Rank1 = validate()

    # Log to TensorBoard

    if cfg.log_to_file:
      if writer is None:
        writer = SummaryWriter(log_dir=osp.join(cfg.exp_dir, 'tensorboard'))
      writer.add_scalars(
        'val scores',
        dict(mAP=mAP,
             Rank1=Rank1),
        ep)

    # save ckpt
    if cfg.log_to_file:
      save_ckpt(modules_optims, step + 1, 0, cfg.ckpt_file)

  ########
  # Test #
  ########

  test(load_model_weight=False)


if __name__ == '__main__':
  main()
