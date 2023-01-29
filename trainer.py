import os
import shutil

import numpy as np
import torch
import torch.nn as nn
from tqdm import trange
from tensorboardX import SummaryWriter
from monai.data import decollate_batch
from torchvision.utils import make_grid
from utils import sdf_normalization, nsdf2prob, coords_like
import torch.nn.functional as F

def train_epoch(
    model, loader, optimizer, losses, device, args):


    model.train()
    for batch_data in loader:
    
        if args.model_name == 'pinns_2d':
            inps = coords_like(batch_data['boundary'], spatial=2).to(device)
            boundary = batch_data['boundary'].to(device)
            label = batch_data['label'].to(device)
            sdf = batch_data['sdf'].to(device)
            gts = {'boundary' : boundary, 'label':label, 'sdf':sdf}
            optimizer.zero_grad()
            outputs = model(inps)

        if args.model_name == 'deeponet_2d':
            inps = coords_like(batch_data['boundary'], spatial=2).to(device)
            boundary = batch_data['boundary'].to(device)
            sdf = batch_data['sdf'].to(device)
            label = batch_data['label'].to(device)
            gts = {'sdf' : sdf, 'boundary' : boundary, 'label': label}
            cnt = batch_data['cnt']
            b, nc, nx, ny = inps.shape
            # only valid for 2 class 
            feats = cnt[:,0,:,:].reshape(b, -1).unsqueeze(-1).unsqueeze(-1).repeat(1,1,nx, ny).to(device)
            optimizer.zero_grad()
            outputs = model(inps, feats)
        
        if args.model_name in [
            'unet_2d', 'fno_2d', 'pino_2d', 'fno_3d-b', 'mixno_2d',
            'unet_3d', 'fno_3d', 'pino_3d', 'unet_3d-b', 'uno_3d-b',
            'uno_2d', 'uno_3d']:

            if '-b' in args.model_name:
                inps = batch_data['boundary'].to(device)
            else:
                inps = batch_data['image'].to(device)

            label = batch_data['label'].to(device)
            boundary = batch_data['boundary'].to(device)
            sdf = batch_data['sdf'].to(device)
            gts = {'label' : label, 'boundary' : boundary, 'sdf': sdf}
            optimizer.zero_grad()

            outputs = model(inps)
            if args.tanh:                
                gts['sdf'] = sdf_normalization(gts['sdf'])
        
        if args.model_name == 'fadeeponet_2d':
            inps = batch_data['image'].to(device)
            label = batch_data['label'].to(device)
            boundary = batch_data['boundary'].to(device)
            sdf = batch_data['sdf'].to(device)
            gts = {'label' : label, 'boundary' : boundary, 'sdf': sdf}
            optimizer.zero_grad()
            outputs = model(inps)
        
        loss = losses.calc_loss(outputs, gts, inps.shape[0])
        
        loss.backward()
        optimizer.step()

def val_epoch(
    model, loader, post_trans, metrics, device, args, sample_idx=3):
    
    model.eval()
    # with torch.no_grad():
    for idx, batch_data in enumerate(loader):
        
        if args.model_name == 'pinns_2d':
            inps = coords_like(batch_data['boundary'], spatial=2).to(device)
            boundary = batch_data['boundary'].to(device)
            label = batch_data['label'].to(device)
            sdf = batch_data['sdf'].to(device)
            gts = {'boundary' : boundary, 'sdf':sdf, 'label':label}
            output = model(inps)
            metrics.calc_metrics(output, gts)
            sample = {'sdf' : output['outputs'], 'boundary': boundary}
            return sample

        if args.model_name == 'deeponet_2d':
            inps = coords_like(batch_data['boundary'], spatial=2).to(device)
            boundary = batch_data['boundary'].to(device)
            sdf = batch_data['sdf'].to(device)
            label = batch_data['label'].to(device)
            gts = {'sdf' : sdf, 'boundary' : boundary, 'label':label}
            cnt = batch_data['cnt']
            b, nc, nx, ny = inps.shape
            # only valid for 2 class 
            feats = cnt[:,0,:,:].reshape(b, -1).unsqueeze(-1).unsqueeze(-1).repeat(1,1,nx, ny).to(device)
            # forward
            output = model(inps, feats)
            metrics.calc_metrics(output, gts)

            if sample_idx == idx:
                sample = {
                    'sdf' : output['outputs'], 'boundary': boundary}

        
        if args.model_name == 'fadeeponet_2d':
            inps = batch_data['image'].to(device)
            label = batch_data['label'].to(device)
            boundary = batch_data['boundary'].to(device)
            sdf = batch_data['sdf'].to(device)
            gts = {'label' : label, 'boundary' : boundary, 'sdf': sdf}
            output = model(inps)
            if args.outsdf:
                output['pred'] = nsdf2prob(output['outputs'])
            post_pred = [post_trans['pred'](i) for i in decollate_batch(output['pred'])]
            post_label = [post_trans['label'](i) for i in decollate_batch(label)]
            
            output['preds'] = post_pred 
            gts['label'] = post_label
            
            metrics.calc_metrics(output, gts)

            if sample_idx == idx:
                sample = {
                    'sdf' : output['outputs'], 'boundary': boundary, 'pred' : post_pred, 'image' : inps, 'gnorm': output['grad_norms']}
        
        if args.model_name in [
            'fno_2d', 'pino_2d', 'unet_2d', 'uno_3d-b', 'mixno_2d',
            'unet_3d', 'fno_3d', 'uno_2d', 'uno_3d', 'unet_3d-b', 'fno_3d-b']:

            if '-b' in args.model_name:
                inps = batch_data['boundary'].to(device)
            else:
                inps = batch_data['image'].to(device)
            label = batch_data['label'].to(device)
            boundary = batch_data['boundary'].to(device)
            sdf = batch_data['sdf'].to(device)
            gts = {'label' : label, 'boundary' : boundary, 'sdf': sdf}
            

            if args.model_name == 'unet_2d':
                grids_lr = coords_like(torch.rand((1, 1, args.low_res, args.low_res)), spatial=2, permute=True).to(device)
                inps_lr = F.grid_sample(inps, grids_lr, mode='bilinear', align_corners=True)
                output = model(inps_lr)
            else:
                output = model(inps)

            if args.model_name == 'unet_2d':
                grids_hr = coords_like(sdf, spatial=2, permute=True).to(inps)
                output['outputs'] = F.grid_sample(output['outputs'], grids_hr, mode='bilinear', align_corners=True)

            if args.outsdf:
                output['pred'] = nsdf2prob(output['outputs'])
            else:
                output['pred'] = output['outputs']
                
            post_pred = [post_trans['pred'](i) for i in decollate_batch(output['pred'])]
            post_label = [post_trans['label'](i) for i in decollate_batch(label)]
            
            output['preds'] = post_pred 
            gts['label'] = post_label
            
            metrics.calc_metrics(output, gts)

            if sample_idx == idx:
                sample = {
                    'sdf' : output['outputs'], 'gt_sdf':sdf, 'boundary': boundary, 'pred' : post_pred, 'image' : inps}
        
    return sample

from utils import draw_pinns2d_result, draw_unet2d_result, draw_unet3d_result

def vis_result(sample, writer, epoch, model_name):

    if model_name == 'pinns_2d':
        pred_sdf = sample['sdf'][0].detach().cpu()
        boundary = sample['boundary'][0].detach().cpu()
        vis_out = draw_pinns2d_result(boundary, pred_sdf, [-0.2, -0.1, 0, 0.1, 0.2])
        nc = pred_sdf.shape[0]
        for ic in range(1, nc):
            writer.add_image('val/pinns_2d/pred_sdf-{:}'.format(ic), vis_out['sdf'][ic-1], epoch)
    
    if model_name == 'deeponet_2d':
        pred_sdf = sample['sdf'][0].detach().cpu()
        boundary = sample['boundary'][0].detach().cpu()
        vis_out = draw_pinns2d_result(boundary, pred_sdf, [-0.2, -0.1, 0, 0.1, 0.2])
        nc = pred_sdf.shape[0]
        for ic in range(1, nc):
            writer.add_image('val/deeponet_2d/pred_sdf-{:}'.format(ic), vis_out['sdf'][ic-1], epoch)
    
    if model_name in ['fadeeponet_2d']:
        image = sample['image'][0].detach().cpu().as_tensor()
        boundary = sample['boundary'][0].detach().cpu().as_tensor()
        pred = sample['pred'][0].detach().cpu().as_tensor()
        sdf = sample['sdf'][0].detach().cpu().as_tensor()
        gnorm = sample['gnorm'][0].detach().cpu().as_tensor()
        err = (gnorm - 1).abs()
        vis_out = draw_unet2d_result(image, pred, boundary, sdf, err, levels= [-.2,-.1,0,.1,.2])
        nc = pred.shape[0]

        for ic in range(1, nc):
            writer.add_image('val/{:}/pred_img-{:}'.format(model_name, ic), vis_out['pred'][ic-1][0], epoch)
            writer.add_image('val/{:}/pred_boundary-{:}'.format(model_name, ic), vis_out['pred'][ic-1][1], epoch)
            writer.add_image('val/{:}/pred_sdf-{:}'.format(model_name, ic), vis_out['sdf'][ic-1], epoch)
            writer.add_image('val/{:}/gnorm_err-{:}'.format(model_name, ic), vis_out['err'][ic-1], epoch)            
    
    if model_name in ['fno_2d', 'unet_2d', 'mixno_2d', 'pino_2d', 'uno_2d']:
        image = sample['image'][0].detach().cpu().as_tensor()
        boundary = sample['boundary'][0].detach().cpu().as_tensor()
        pred = sample['pred'][0].detach().cpu().as_tensor()
        sdf = sample['sdf'][0].detach().cpu().as_tensor()
        gt_sdf = sample['gt_sdf'][0].detach().cpu().as_tensor()
        err = (sdf - gt_sdf).abs()
        vis_out = draw_unet2d_result(image, pred, gt_sdf>0, sdf, err, levels= [-.2,-.1,0,.1,.2], err_rng=[0, 0.03])
        nc = pred.shape[0]

        for ic in range(1, nc):
            writer.add_image('val/{:}/pred_img-{:}'.format(model_name, ic), vis_out['pred'][ic-1][0], epoch)
            writer.add_image('val/{:}/pred_boundary-{:}'.format(model_name, ic), vis_out['pred'][ic-1][1], epoch)
            writer.add_image('val/{:}/pred_sdf-{:}'.format(model_name, ic), vis_out['sdf'][ic-1], epoch)
            writer.add_image('val/{:}/op_err-{:}'.format(model_name, ic), vis_out['err'][ic-1], epoch)
    
    if model_name in ['fno_3d', 'unet_3d', 'pino_3d', 'uno_3d', 'fno_3d-b', 'unet_3d-b', 'uno_3d-b']:
        image = sample['image'][0].detach().cpu().as_tensor()
        boundary = sample['boundary'][0].detach().cpu().as_tensor()
        pred = sample['pred'][0].detach().cpu().as_tensor()
        sdf = sample['sdf'][0].detach().cpu().as_tensor()
        gt_sdf = sample['gt_sdf'][0].detach().cpu().as_tensor()
        err = (sdf - gt_sdf).abs()
        vis_out = draw_unet3d_result(image, pred, boundary, sdf, err, alpha=0.4, levels= [-.2,-.1,0,.1,.2], err_rng=[0, 0.1])
        nc = pred.shape[0]

        for ic in range(1, nc):
            writer.add_image('val/{:}/pred_img-{:}'.format(model_name, ic), vis_out['pred'][ic][0], epoch)
            writer.add_image('val/{:}/pred_boundary-{:}'.format(model_name, ic), vis_out['pred'][ic][1], epoch)
            writer.add_image('val/{:}/pred_sdf-{:}'.format(model_name, ic), vis_out['sdf'][ic], epoch)
            writer.add_image('val/{:}/op_err-{:}'.format(model_name, ic), vis_out['err'][ic], epoch)
    

def save_checkpoint(
    model, epoch, out_dir, filename,
    best_metrics, optimizer, scheduler):

    state_dict = model.state_dict()
    save_dict = {
        "epoch": epoch, "best_metrics": best_metrics, "state_dict": state_dict}
   
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()

    filename = os.path.join(out_dir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)

def run_training(
    model, 
    train_loader, 
    val_loader,
    optimizer,
    train_losses,
    eval_metrics,
    post_trans,
    args,
    scheduler,
    start_epoch):

    writer = None 
    if args.logdir is not None:
        writer = SummaryWriter(logdir=args.logdir)
        print("Writing Tensorboard logs to ", args.logdir)

    device = torch.device(args.device)
    model.to(device)

    for epoch in trange(start_epoch, args.max_epochs):
        train_epoch(model, train_loader, optimizer, train_losses, device, args)
        train_losses.update_losses(writer, epoch)
        train_losses.disp_losses()
        train_losses.reset_losses()

        if ((epoch + 1) % args.val_every == 0) & (epoch > args.warmup_epochs):
            sample = val_epoch(
                model, val_loader, post_trans, eval_metrics, device, args)

            if args.model_name in ['pinns_2d', 'deeponet_2d']:
                is_update = eval_metrics.update_metrics(writer, epoch, 'PDEError', False)
            elif args.model_name in ['fadeeponet_2d', 'unet_2d', 'mixno_2d', 'uno_2d', 'fno_2d']:
                is_update = eval_metrics.update_metrics(writer, epoch, 'DiceMetric', True)
            elif args.model_name in [
                'pino_2d', 'fno_3d-b', 'unet_3d-b', 'uno3d-b',
                'unet_3d', 'fno_3d', 'uno_3d']:
                is_update = eval_metrics.update_metrics(writer, epoch, 'OPError', False)
            
            eval_metrics.disp_metrics()
            eval_metrics.reset_metrics()
            if is_update:
                save_checkpoint(
                    model, epoch, args.logdir, 
                    'best.pt', eval_metrics.metric_best, 
                    optimizer, scheduler)
                print('save new best model')
                vis_result(sample, writer, epoch, args.model_name)
                print('save new best result visualization')
            else:
                save_checkpoint(
                    model, epoch, args.logdir, 
                    'latest.pt', eval_metrics.metric_best, 
                    optimizer, scheduler)
                print('save latest model')

        scheduler.step()
        writer.add_scalar("lr", scheduler.get_last_lr(), epoch)

    print("Training Finished ")

def run_evaluation(
    model, 
    val_loader,
    eval_metrics,
    post_trans,
    args,
    model_inferer):

    device = torch.device(args.device)
    model.to(device)
    val_epoch(model, val_loader, post_trans, eval_metrics, device, model_inferer, args)
    eval_metrics.update_metrics()
    eval_metrics.disp_metrics()

