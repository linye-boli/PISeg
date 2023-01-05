from monai.metrics import (
    DiceMetric, 
    SurfaceDistanceMetric, 
    HausdorffDistanceMetric, )
from tabulate import tabulate
from copy import deepcopy
import torch.nn as nn

from losses import EikBoundaryLoss, EikPDELoss, FMMLoss
from monai.metrics import LossMetric 

bcloss = EikBoundaryLoss()
pdeloss = EikPDELoss(g=1, is_std=False)
fmmloss = FMMLoss()

class Metrics:
    def __init__(self, cfgs):
        metric_dict = {}
        metric_best = {'epoch':0}
        print('metric for testing : ')
        for metric_nm, cfg in cfgs.items():
            if metric_nm == 'PDEError':
                metric_dict[metric_nm] = LossMetric(loss_fn=pdeloss)
            elif metric_nm == 'BCError':
                metric_dict[metric_nm] = LossMetric(loss_fn=bcloss)
            elif metric_nm == 'OPError':
                metric_dict[metric_nm] = LossMetric(loss_fn=fmmloss)
            else:
                metric_dict[metric_nm]=eval(metric_nm)(**cfg['kargs'])

            print('nm : ', metric_nm)
            if metric_nm == 'DiceMetric':
                metric_best[metric_nm] = 0.0
            elif metric_nm == 'SurfaceDistanceMetric':
                metric_best[metric_nm] = 1000
            elif metric_nm == 'HausdorffDistanceMetric':
                metric_best[metric_nm] = 1000
            elif metric_nm == 'PDEError':
                metric_best[metric_nm] = 1000
            elif metric_nm == 'BCError':
                metric_best[metric_nm] = 1000
            elif metric_nm == 'OPError':
                metric_best[metric_nm] = 1000
                        
        self.metric_dict = metric_dict 
        self.metric_best = metric_best 
        self.metric_cur = deepcopy(metric_best)

    def reset_metrics(self):
        for _, metric in self.metric_dict.items():
            metric.reset()
    
    def update_metrics(self, writer=None, epoch=None, mname='DiceMetric', greater=True):
        self.metric_cur['epoch'] = epoch    

        for nm, metric in self.metric_dict.items():
            val = metric.aggregate().item()
            self.metric_cur[nm] = val

            if writer is not None:
                writer.add_scalar(nm, val, epoch)
        
        if greater:
            if self.metric_cur[mname] > self.metric_best[mname]:
                self.metric_best = deepcopy(self.metric_cur)
                return True 
            else:
                return False
        else:
            if self.metric_cur[mname] < self.metric_best[mname]:
                self.metric_best = deepcopy(self.metric_cur)
                return True 
            else:
                return False

    def disp_metrics(self):
        headers = []
        values = []
        for nm, val in self.metric_cur.items():
            headers.append(nm)
            values.append(val)
        
        for nm, val in self.metric_best.items():
            headers.append('best-'+nm)
            values.append(val)
        
        print(tabulate([values], headers=headers, tablefmt='fancy_grid'))

    def calc_metrics(self, preds, gts):
        for metric_nm, metric in self.metric_dict.items():
            if metric_nm == 'PDEError':
                metric(preds['grad_norms'])
            elif metric_nm == 'BCError':
                metric(preds['outputs'], gts['boundary'])
            elif metric_nm == 'OPError':
                metric(preds['outputs'], gts['sdf'])
            else:
                metric(preds['preds'], gts['label'])
        