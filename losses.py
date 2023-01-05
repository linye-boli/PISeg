import torch
import torch.nn as nn
from monai.networks import one_hot
from monai.losses import DiceLoss
from monai.losses import GeneralizedDiceLoss
from torch.nn import L1Loss

class BoundaryLoss(nn.Module):
    '''
    Kervadec, Hoel, et al. "Boundary loss for highly unbalanced segmentation." International conference on medical imaging with deep learning. PMLR, 2019.
    '''
    def __init__(self, idc, k, softmax=True):
        super(BoundaryLoss, self).__init__()
        self.idc = idc
        self.k = k
        self.softmax=softmax
    
    def forward(self, preds, sdfs, labels=None):
        if self.softmax:
            probs = torch.softmax(preds, 1)
        else:
            probs = preds
        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = sdfs[:, self.idc, ...].type(torch.float32)

        if labels is None:
            loss = (dc * pc).mean()
            return loss
        else:
            gts = one_hot(labels, self.k, dim=1)
            gt = gts[:, self.idc, ...]
            loss = (dc * (pc - gt)).mean()
            return loss

from utils import sdf_normalization, nsdf2prob
class SDMProdLoss(nn.Module):
    '''
    Xue, Yuan, et al. "Shape-aware organ segmentation by predicting signed distance maps." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 34. No. 07. 2020.
    '''
    def __init__(self, smooth=1e-5):
        super(SDMProdLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred_nsdfs, gt_nsdfs):
        intersect = gt_nsdfs * pred_nsdfs
        pd_sum = pred_nsdfs ** 2
        gt_sum = gt_nsdfs ** 2
        loss = -((intersect + self.smooth) / (intersect + pd_sum + gt_sum + self.smooth)).mean()
        return loss 

class SDFDiceLoss(nn.Module):
    def __init__(self, to_prob=True, to_onehot_y=True, softmax=False):
        super(SDFDiceLoss, self).__init__()
        self.to_prob = to_prob 
        self.dice = DiceLoss(to_onehot_y=to_onehot_y, softmax=softmax)
    
    def forward(self, pred_sdf, label):
        if self.to_prob:
            pred = torch.sigmoid(pred_sdf * 1500)
        else:
            pred = pred_sdf 
        
        return self.dice(pred, label)

import torch.nn.functional as F
class FMMLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(FMMLoss, self).__init__()
        self.reduction = reduction 

    def forward(self, pred_sdf, fmm_sdf):
        loss = (pred_sdf - fmm_sdf).abs()
        return loss.mean()


class SchemeLoss(nn.Module):
    def __init__(self, spatial=2):
        super(SchemeLoss, self).__init__()
        self.spatial = spatial

    def forward(self, u):
        if self.spatial == 2:
            b, nc, nx, ny = u.shape

            dx = 2 / (nx - 1)
            dy = 2 / (ny - 1)

            u_center = u[:,:,1:-1,1:-1]
            u_left = u[:,:,1:-1,:-2]
            u_right = u[:,:,1:-1,2:]
            u_up = u[:,:,:-2,1:-1]
            u_down = u[:,:,2:,1:-1]

            u_xmax = torch.relu(torch.max(u_center - u_left, u_center - u_right)) / dx  #godunov
            u_ymax = torch.relu(torch.max(u_center - u_up, u_center - u_down)) / dy

            loss = ((u_xmax + u_ymax)**0.5 - 1).abs()
            return loss.mean()

from monai.networks import one_hot 
class SignLoss(nn.Module):
    def __init__(self, reduction='sum'):
        super(SignLoss, self).__init__()
        self.reduction = reduction 
    
    def forward(self, pred_sdf, label, boundary):
        num_cat = len(label.unique())
        sign = (one_hot(label, num_cat) - 0.5) * 2
        sign = sign - sign * boundary

        if self.reduction == 'mean':
            loss = torch.relu(pred_sdf * sign).mean()
        elif self.reduction == 'sum':
            loss = torch.relu(pred_sdf * sign).sum()

        return loss

class EikBoundaryLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(EikBoundaryLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, pred_sdf, boundary):
        loss = (pred_sdf * boundary).abs()
        if self.reduction == 'mean':
            if boundary.sum() > 0:
                return loss.sum() / boundary.sum()
            else:
                return loss.sum().mean()

class EikPDELoss(nn.Module):
    def __init__(self, g=1, is_std=True):
        super(EikPDELoss, self).__init__()
        self.g = g
        self.is_std = is_std
    
    def forward(self, grad_norms):
        loss_mean = (grad_norms - self.g).abs().mean()

        if self.is_std:
            loss_std = grad_norms.std()
            return loss_mean + loss_std
        else:
            return loss_mean  

from utils import AverageMeter
from tabulate import tabulate

class Losses:
    def __init__(self, cfgs, rebalance=False, outsdf=False):
        loss_dict = {}
        loss_cur = {'epoch':0}
        print('loss for training : ')
        for loss_nm, cfg in cfgs.items():
            loss_dict[loss_nm]={
                    'func' : eval(loss_nm)(**cfg['kargs']),
                    'w' : cfg['w'],
                    'logger' : AverageMeter()}
            loss_cur[loss_nm]=100
            print('nm :', loss_nm, ' w :', cfg['w'])
        self.loss_dict = loss_dict
        self.loss_cur = loss_cur
        self.rebalance = rebalance
        self.outsdf = outsdf
        self.loss_nms = list(loss_dict.keys())

    def reset_losses(self):
        for _, loss in self.loss_dict.items():
            loss['logger'].reset()
    
    def disp_losses(self):
        headers = []
        values = []
        for nm, val in self.loss_cur.items():
            headers.append(nm)
            values.append(val)
        print('\n')
        print(tabulate([values], headers=headers, tablefmt='simple_grid'))
    
    def update_losses(self, writer, epoch):
        self.loss_cur['epoch'] = epoch
        loss_sum = 0
        for nm, loss in self.loss_dict.items():
            loss_val = loss['logger'].avg
            loss_sum += loss_val * loss['w']
            self.loss_cur[nm] = loss_val
            writer.add_scalar(nm, loss_val, epoch)
        writer.add_scalar('loss-all', loss_sum, epoch)
        
        if (self.rebalance) & ('BoundaryLoss' in self.loss_nms):
            if 'DiceLoss' in self.loss_nms:
                if self.loss_dict['DiceLoss']['w'] > 0.01:
                    self.loss_dict['BoundaryLoss']['w'] += 0.001
                    self.loss_dict['DiceLoss']['w'] = 1 - self.loss_dict['BoundaryLoss']['w']
            
                print('\n')
                print("BoundaryLoss w : {:.4f}".format(self.loss_dict['BoundaryLoss']['w']))
                print("DiceLoss w : {:.4f}".format(self.loss_dict['DiceLoss']['w']))
                print('\n')
            
            if 'GeneralizedDiceLoss' in self.loss_nms:
                if self.loss_dict['GeneralizedDiceLoss']['w'] > 0.01:
                    self.loss_dict['BoundaryLoss']['w'] += 0.001
                    self.loss_dict['GeneralizedDiceLoss']['w'] = 1 - self.loss_dict['BoundaryLoss']['w']
            
                print('\n')
                print("BoundaryLoss w : {:.4f}".format(self.loss_dict['BoundaryLoss']['w']))
                print("GeneralizedDiceLoss w : {:.4f}".format(self.loss_dict['GeneralizedDiceLoss']['w']))
                print('\n')
        
        if (self.rebalance) & ('EikBoundaryLoss' in self.loss_nms) & ('EikPDELoss' in self.loss_nms):# & ('SDFDiceLoss' in self.loss_nms):
            # if self.loss_dict['SDFDiceLoss']['w'] < 1:
            #     self.loss_dict['SDFDiceLoss']['w'] += 0.1
            
            if self.loss_dict['EikPDELoss']['w'] > 1:
                self.loss_dict['EikPDELoss']['w'] -= .01
            
            # if self.loss_dict['EikBoundaryLoss']['w'] < 10:
            #     self.loss_dict['EikBoundaryLoss']['w'] += .1

            print('\n')
            print("EikPDELoss w : {:.4f}".format(self.loss_dict['EikPDELoss']['w']))
            print("EikBoundaryLoss w : {:.4f}".format(self.loss_dict['EikBoundaryLoss']['w']))
            # print("SDFDiceLoss w : {:.4f}".format(self.loss_dict['SDFDiceLoss']['w']))
            print('\n')

    def calc_loss(self, outputs, gts, bs):
        loss_out = 0
        for nm, loss in self.loss_dict.items():
            if (nm == 'DiceLoss') or (nm == 'GeneralizedDiceLoss'):
                if self.outsdf:
                    diceloss = loss['func'](
                        nsdf2prob(outputs['outputs']), gts['label'])
                else:
                    diceloss = loss['func'](outputs['outputs'], gts['label'])
                loss['logger'].update(diceloss.item(), n=bs)
                loss_out += diceloss * loss['w']
            if nm == 'BoundaryLoss':
                boundaryloss = loss['func'](outputs['outputs'], gts['sdf'])#, gts['label'])
                loss['logger'].update(boundaryloss.item(), n=bs)
                loss_out += boundaryloss * loss['w']
            if nm == 'SDMProdLoss':
                sdmloss = loss['func'](outputs['outputs'], gts['sdf'])
                loss['logger'].update(sdmloss.item(), n=bs)
                loss_out += sdmloss * loss['w']
            if nm == 'L1Loss':
                l1loss = loss['func'](outputs['outputs'], gts['sdf'])
                loss['logger'].update(l1loss.item(), n=bs)
                loss_out += l1loss * loss['w']
            if nm == 'EikPDELoss':
                eikpdeloss = loss['func'](outputs['grad_norms'])#, gts['sdf'])
                loss['logger'].update(eikpdeloss.item(), n=bs)
                loss_out += eikpdeloss * loss['w']
            if nm == 'EikBoundaryLoss':
                eikboundaryloss = loss['func'](outputs['outputs'], gts['boundary'])
                loss['logger'].update(eikboundaryloss.item(), n=bs)
                loss_out += eikboundaryloss * loss['w']
            if nm == 'SDFDiceLoss':
                sdfdiceloss = loss['func'](outputs['outputs'], gts['label'])
                loss['logger'].update(sdfdiceloss.item(), n=bs)
                loss_out += sdfdiceloss * loss['w']
            if nm == 'SignLoss':
                signloss = loss['func'](outputs['outputs'], gts['label'], gts['boundary'])
                loss['logger'].update(signloss.item(), n=bs)
                loss_out += signloss * loss['w']
            if nm == 'SchemeLoss':
                schemeloss = loss['func'](outputs['outputs'])
                loss['logger'].update(schemeloss.item(), n=bs)
                loss_out += schemeloss * loss['w']
            if nm == 'FMMLoss':
                fmmloss = loss['func'](outputs['outputs'], gts['sdf'])
                loss['logger'].update(fmmloss.item(), n=bs)
                loss_out += fmmloss * loss['w']
        
        return loss_out
    
