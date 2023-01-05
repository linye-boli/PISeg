import argparse
import yaml 
import os 
import random 
import numpy as np
from functools import partial

import torch
import torch.backends.cudnn as cudnn
from monai.inferers import sliding_window_inference
from monai.optimizers.lr_scheduler import WarmupCosineSchedule
from torch.optim.lr_scheduler import MultiStepLR

from data_utils import (
    get_deeponet2d_loader, 
    get_pinns2d_loader,
    get_pinns3d_loader,
    get_unet2d_loader,
    get_unet3d_loader,
    post_transform
)

from network import (
    PINNS,
    UNet, 
    DeepONet,
    CNNDeepONet,
    FADeepONet,
    FNO,
    PINO
)

from losses import Losses
from metrics import Metrics
from trainer import run_training, run_evaluation

parser = argparse.ArgumentParser(description='segop pipeline')
parser.add_argument(
    "--checkpoint", default=None, type=str, help="start training from saved checkpoint")
parser.add_argument(
    "--logdir", default="test", type=str, help="directory to save the tensorboard logs")
parser.add_argument(
    "--device", default="cuda:0", type=str, help="directory to save the tensorboard logs")
parser.add_argument(
    "--traindata_dir", default="/dataset/MSD/SpleenPreprocess/", type=str, help="dataset directory")
parser.add_argument(
    "--valdata_dir", default="/dataset/MSD/SpleenPreprocess/", type=str, help="dataset directory")
parser.add_argument(
    "--sample_idx", default=0, type=int, help="data sample for pinns")
parser.add_argument(
    "--is_sample", default=0, type=int, help="data sample for pinns")
parser.add_argument(
    "--num_cat", default=None, type=int, help="number of category including bg")
parser.add_argument(
    "--max_epochs", default=1000, type=int, help="max number of training epochs")
parser.add_argument(
    "--batch_size", default=8, type=int, help="number of batch size")
parser.add_argument(
    "--optim_lr", default=1e-4, type=float, help="optimization learning rate")
parser.add_argument(
    "--val_every", default=10, type=int, help="validation frequency")
parser.add_argument(
    "--val_only", default=0, type=int, help="only evaluation")
parser.add_argument(
    "--workers", default=8, type=int, help="number of workers")
parser.add_argument(
    "--model_name", default="unet", type=str, help="model name")
parser.add_argument(
    "--tanh", default=0, type=int, help="whether use tanh as final out layer")
parser.add_argument(
    "--outsdf", default=0, type=int, help="whether model output is sdf")
parser.add_argument(
    "--loss_cfg", default=None, type=str, help="loss configuration")
parser.add_argument(
    "--metric_cfg", default=None, type=str, help="metric configuration")
parser.add_argument(
    "--rebalance", default=0, type=int, help="loss rebalance strategy")
parser.add_argument(
    "--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
parser.add_argument(
    "--warmup_epochs", default=50, type=int, help="number of warmup epochs")
parser.add_argument(
    '--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument(
    '--seed', type=int,  default=2019, help='random seed')

def main():
    args = parser.parse_args()
    if args.model_name == 'pinns_2d':
        args.logdir = os.path.join(
            args.logdir, args.model_name, str(args.sample_idx), 'seed_{:}'.format(args.seed))
    if args.model_name == 'deeponet_2d':
        args.logdir = os.path.join(
            args.logdir, args.model_name, 'seed_{:}'.format(args.seed))
    # if args.model_name == 'unet_2d':
    #     args.logdir = os.path.join(
    #         args.logdir, args.model_name, 'seed_{:}'.format(args.seed))
    if args.model_name == 'fadeeponet_2d':
        args.logdir = os.path.join(
            args.logdir, args.model_name, 'seed_{:}'.format(args.seed))
    if args.model_name in ['fno_2d', 'unet_2d', 'pino_2d']:
        args.logdir = os.path.join(
            args.logdir, args.model_name, 'seed_{:}'.format(args.seed))
    

    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed) 

    # load dataset 
    if args.model_name == 'pinns_2d':
        train_loader, val_loader = get_pinns2d_loader(
            data_dir = args.data_dir, 
            train_batchsize = args.batch_size, 
            num_workers = args.workers, 
            sample_idx=args.sample_idx)
    if args.model_name == 'deeponet_2d':
        train_loader, val_loader = get_deeponet2d_loader(
            data_dir = args.data_dir,
            train_batchsize=args.batch_size,
            num_workers = args.workers)
    if args.model_name in ['unet_2d', 'fno_2d', 'pino_2d']:
        train_loader, val_loader = get_unet2d_loader(
            traindata_dir = args.traindata_dir,
            valdata_dir = args.valdata_dir, 
            train_batchsize=args.batch_size,
            num_workers = args.workers)
    if args.model_name == 'fadeeponet_2d':
        train_loader, val_loader = get_unet2d_loader(
            data_dir = args.data_dir,
            train_batchsize=args.batch_size,
            num_workers = args.workers,
            sample=args.is_sample)

    post_trans = post_transform(args.num_cat)

    # build loss and metrics
    loss_config = yaml.load(open(args.loss_cfg, 'r'), Loader=yaml.Loader)
    if 'BoundaryLoss' in loss_config.keys():
        loss_config['BoundaryLoss']['kargs']['idc'] = [i for i in range(args.num_cat)]
        loss_config['BoundaryLoss']['kargs']['k'] = args.num_cat

    metric_config = yaml.load(open(args.metric_cfg, 'r'), Loader=yaml.Loader)

    train_losses = Losses(loss_config, rebalance=args.rebalance, outsdf=args.outsdf)
    eval_metrics = Metrics(metric_config)

    # build model
    if args.model_name == 'pinns_2d':
        model = PINNS(spatial_dim=2, out_channels=args.num_cat)
    if args.model_name == 'deeponet_2d':
        model = DeepONet(spatial_dim=2, out_channels=args.num_cat, feat_dim=100)
    if args.model_name == 'unet_2d':
        model = UNet(spatial_dims=2, out_channels=args.num_cat, tanh=args.tanh)
    if args.model_name == 'fadeeponet_2d':
        model = FADeepONet(spatial_dims=2, out_channels=args.num_cat)
    if args.model_name == 'fno_2d':
        model = FNO(spatial_dims=2, out_channels=args.num_cat)
    if args.model_name == 'pino_2d':
        model = PINO(spatial_dims=2, out_channels=args.num_cat)
    

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)

    # build optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr)    
    if args.lrschedule == 'warmup_cosine':
        scheduler = WarmupCosineSchedule(
            optimizer, warmup_steps=args.warmup_epochs, t_total=args.max_epochs)
    elif args.lrschedule == 'multistep':
        scheduler = MultiStepLR(optimizer, milestones=[5000, 8000], gamma=0.1)

    start_epoch = 0
    best_metrics = 0
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        best_metrics = checkpoint["best_metrics"]
        print("=> loaded checkpoint '{}' (epoch {})".format(args.checkpoint, start_epoch))
        print(best_metrics)

    if args.val_only:
        run_evaluation(
            model, val_loader, eval_metrics, post_trans, args
        )
    else:
        run_training(
            model, 
            train_loader, 
            val_loader,
            optimizer,
            train_losses,
            eval_metrics,
            post_trans,
            args,
            scheduler,
            start_epoch)

if __name__ == '__main__':
    main()