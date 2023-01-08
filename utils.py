import numpy as np

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)

import torch
from torchvision.utils import make_grid
from torchvision.utils import draw_segmentation_masks
from monai.networks import one_hot

COLORS = [
    (0,0,0), # bg
    (255,0,0), # red
    (0,255,0), # lime
    (0,0,255), # blue
    (255,255,0),# yellow
    (0,255,255),# cyan
    (255,0,255) # magenta
]

COLORS_ = [
    'k', 'r', 'g', 'b', 'y', 'c', 'm'
]

def select_slice(label, num_slice=10):
    z_rng = torch.argwhere(label[0]!=0)[:,2]
    z_min, z_max = z_rng.min(), z_rng.max()
    idx = torch.linspace(z_min, z_max, num_slice).int().tolist()
    return idx

def draw_result(image, label, pred, num_cat, num_slice=20, num_row=5, alpha=0.2):
    # image : 1xyz
    # label : cxyz
    slice_idx = select_slice(label, num_slice)
    image = image[:,:,:,slice_idx].permute(3,0,1,2)
    label = label[:,:,:,slice_idx].permute(3,0,1,2).astype(torch.bool)
    pred = pred[:,:,:,slice_idx].permute(3,0,1,2).astype(torch.bool)
    
    image_grid = make_grid(image, 5, normalize=False)
    image_grid = (image_grid * 255).astype(torch.uint8)
    
    label_grid = make_grid(label, num_row, normalize=False)
    gt_result = draw_segmentation_masks(image_grid, label_grid, alpha, colors=COLORS[:num_cat])
    
    pred_grid = make_grid(pred, num_row, normalize=False)
    pred_result = draw_segmentation_masks(image_grid, pred_grid, alpha, colors=COLORS[:num_cat])
    
    return gt_result, pred_result

def sdf_normalization(sdf):
    bsize, csize = sdf.size(0), sdf.size(1)
    normalized_sdf = torch.zeros_like(sdf)
    for b in range(bsize):
        for c in range(csize):
            pos_sdf = sdf[b,c].clone()
            pos_sdf[pos_sdf <= 0] = 0
            neg_sdf = sdf[b,c].clone()
            neg_sdf[neg_sdf > 0] = 0 
            if neg_sdf.min() == 0:
                nsdf = pos_sdf/pos_sdf.max()
            elif pos_sdf.max() == 0:
                nsdf = neg_sdf/neg_sdf.min().abs()
            else:
                nsdf = neg_sdf/neg_sdf.min().abs() + pos_sdf/pos_sdf.max()
            normalized_sdf[b,c] = nsdf
    
    return normalized_sdf

def nsdf2prob(nsdf, k=1500):
    return torch.sigmoid(k * nsdf)

from skimage import measure
def levelset2boundary(ls):
    boundary = np.zeros_like(ls)
    num_slice = ls.shape[-1]
    for s in range(num_slice):
        slice = ls[:,:,s]
        if slice.max() > 0:
            bslice = np.zeros_like(slice)
            cnts = measure.find_contours(slice, 0) # zero level set is boundary
            for cnt in cnts:
                cnt = np.around(cnt).astype('int')
                y = cnt[:,0]
                x = cnt[:,1]
                bslice[y, x] = 1
            boundary[:,:,s] = bslice 
    return boundary

from geomdl import fitting 
def cnt_resample(pts, delta = 0.01):
    curve = fitting.interpolate_curve(pts, degree=3)
    curve.evaluate(start=0, stop=1.)
    curve.delta = delta
    cnts_pts = np.array(curve.evalpts)
    return cnts_pts

def levelset2boundary2D(ls, pts=True):
    cnts = measure.find_contours(ls, 0)
    boundary = np.zeros_like(ls)   
    idx = np.argmax([len(cnt) for cnt in cnts])
    cnt = cnts[idx]
    cnt = np.rint(cnt).astype('int')
    y = cnt[:,0]
    x = cnt[:,1]
    boundary[y, x] = 1
    if pts:
        cnt_pts = cnt_resample(cnts[idx].tolist(), 0.02)
        return boundary, cnt_pts
    else:
        return boundary

from kornia.utils import create_meshgrid3d, create_meshgrid
import torch
import torch.nn.functional as F

def nzxyc2ncxyz(volume):
    #01234-04231
    return volume.permute(0,4,2,3,1)

def coords_like(x, spatial=3, permute=False):

    if spatial == 3:
        nb, _, nx, ny, nz = x.shape
        coords = nzxyc2ncxyz(create_meshgrid3d(nz,nx,ny)).repeat(nb,1,1,1,1)
        if permute:
            return coords.permute(0,2,3,4,1).to(x)
        else:
            return coords.to(x)
    elif spatial == 2:
        nb, _, nx, ny = x.shape
        coords = create_meshgrid(nx, ny).repeat(nb, 1, 1, 1)
        if permute:
            return coords.to(x)
        else:
            return coords.permute(0,3,1,2).to(x)

from torch.autograd import grad
def gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    return points_grad

def gradient_norm(y, u):
    # g = gradient(y, u)
    # gnorm = (g**2).sum(axis=1, keepdim=True)
    gnorms = []
    num_cls = u.shape[1]
    for c in range(num_cls):
        g = gradient(y, u[:,[c]])
        gnorm = (g**2).sum(axis=1, keepdim=True) ** 0.5
        gnorms.append(gnorm)
    gnorms = torch.cat(gnorms, axis=1)
    return gnorms

def sample_feats(feats, num_sample=100000, spatial_dim=3, mode='bilinear', is_grid=False):
    if spatial_dim == 3:
        b, nc, nx, ny, nz = feats.shape
        x = ((torch.rand((b, num_sample, 1, 1, 3)) - 0.5)*2).to(feats)
        c = F.grid_sample(feats, x, mode, align_corners=True)    
        x = x.reshape(-1,3)
        c = c.permute(0,2,3,4,1).reshape(-1, nc)
    elif spatial_dim == 2:
        if is_grid:
            b, nc, nx, ny = feats.shape
            x = coords_like(feats, spatial=2, permute=True).to(feats)
            c = F.grid_sample(feats, x, mode, align_corners=False)
            x = x.reshape(-1,2)
            c = c.permute(0,2,3,1).reshape(-1, nc)
        else:
            b, nc, nx, ny = feats.shape
            x = ((torch.rand((b, num_sample, 1, 2)) - 0.5)*2).to(feats)
            c = F.grid_sample(feats, x, mode, align_corners=True)    
            x = x.reshape(-1,2)
            c = c.permute(0,2,3,1).reshape(-1, nc)

    return x, c

def grid_gradient_norm(u, x, spatial = 2):
    if spatial == 3:
        b, nc, nx, ny, nz = u.shape
        g = torch.ones((b, 1, nx, ny, nz), requires_grad=False, device=u.device)
    elif spatial == 2:
        b, nc, nx, ny = u.shape
        g = torch.ones((b, 1, nx, ny), requires_grad=False, device=u.device)

    gnorms = []
    for c in range(nc):
        gnorm = torch.autograd.grad(
            outputs=u[:,[c]], inputs=x, grad_outputs=g, 
            create_graph=True, retain_graph=True, only_inputs=True)[0]
        gnorms.append((gnorm**2).sum(axis=1, keepdim=True))

    return torch.cat(gnorms, axis=1)

from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def draw_pinns2d_result(boundary, ls, levels=[-0.5, -0.1, 0, 0.1, 0.5]):
    nc, ny, nx = boundary.shape
    x, y = np.linspace(-1,1,num=nx), np.linspace(-1,1,num=ny)
    X, Y = np.meshgrid(x, y)

    im_lst = []
    for ic in range(1, nc):
        fig = Figure(figsize=(4,4), dpi=100)
        fig.subplots_adjust(0,0,1,1)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot()
        ax.imshow(boundary[ic], cmap='gray', extent=(-1,1,1,-1))
        CS = ax.contour(X, Y, ls[ic,:,:], levels=levels, colors=COLORS_[ic])
        ax.clabel(CS, inline=True)
        ax.axis('off')
        canvas.draw()
        im = Image.fromarray(np.asarray(canvas.buffer_rgba())).convert('RGB')
        plt.close(fig)
        im_lst.append(PILToTensor()(im)) 
    
    return {'sdf' : im_lst}

def draw_unet2d_result(img, pred, boundary, ls=None, err=None, alpha=0.2, levels=[-0.5, -0.1, 0, 0.1, 0.5]):
    nc, ny, nx = boundary.shape
    x, y = np.linspace(-1,1,num=nx), np.linspace(-1,1,num=ny)
    X, Y = np.meshgrid(x, y)

    
    img = (img * 255).repeat(3,1,1).to(torch.uint8)

    pred_im = []
    for ic in range(1, nc):
        mask = pred[[ic]].to(torch.bool)
        bimg = (boundary[[ic]] * 255).repeat(3,1,1).to(torch.uint8)
        pred_i = draw_segmentation_masks(img, mask, colors=COLORS[ic], alpha=alpha)
        # pred_i = pred_i.permute(1,2,0)
        pred_b = draw_segmentation_masks(bimg, mask, colors=COLORS[ic], alpha=alpha)
        # pred_b = pred_b.permute(1,2,0)
        pred_im.append((pred_i, pred_b))

    if ls is not None:
        sdf_im = []
        for ic in range(1, nc):
            fig = Figure(figsize=(4,4), dpi=100)
            fig.subplots_adjust(0,0,1,1)
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_subplot()
            ax.imshow(boundary[ic], cmap='gray', extent=(-1,1,1,-1))
            CS = ax.contour(X, Y, ls[ic,:,:], levels=levels, colors=COLORS_[ic])
            ax.clabel(CS, inline=True)
            ax.axis('off')
            canvas.draw()
            im = Image.fromarray(np.asarray(canvas.buffer_rgba()))
            plt.close(fig)
            sdf_im.append(PILToTensor()(im)) 

    if err is not None:
        err_im = []
        for ic in range(1, nc):
            fig = Figure(figsize=(4,4), dpi=100)
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_subplot()
            im = ax.imshow(err[ic], cmap='jet', extent=(-1,1,1,-1), vmin=0, vmax=0.03)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='2%', pad=0.1)
            cbar = fig.colorbar(im, cax=cax, orientation='vertical')
            cbar.formatter.set_powerlimits((0, 0))
            cbar.formatter.set_useMathText(True)
            canvas.draw()
            im = Image.fromarray(np.asarray(canvas.buffer_rgba()))
            plt.close(fig)
            err_im.append(PILToTensor()(im)) 
    
    vis_out = {'pred': pred_im}
    if ls is not None:
        vis_out['sdf'] = sdf_im
    
    if err is not None:
        vis_out['err'] = err_im
    return vis_out


from torchvision.transforms import PILToTensor, ToPILImage

def draw_pinns3d_result(boundary, ls, num_slice=10, levels=[-0.5, -0.1, 0, 0.1, 0.5]):
    nc, nx, ny, nz = boundary.shape

    sdf3d = {}
    for ic in range(1, nc):
        slice_idx = select_slice(boundary[[ic]], num_slice = num_slice)

        slice_im = []
        for s in slice_idx:
            sb = boundary[ic, :, :, s]
            sl = ls[ic, :, :, s]

            x, y = np.linspace(-1,1,num=nx), np.linspace(-1,1,num=ny)
            X, Y = np.meshgrid(x, y)

            fig = Figure(figsize=(2,2), dpi=100)
            fig.subplots_adjust(0,0,1,1)
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_subplot()
            ax.imshow(sb, cmap='gray', extent=(-1,1,1,-1))
            CS = ax.contour(X, Y, sl, levels=levels, colors=COLORS_[ic])
            ax.clabel(CS, inline=True)
            ax.axis('off')
            canvas.draw()
            im = Image.fromarray(np.asarray(canvas.buffer_rgba()))
            im = PILToTensor()(im)
            slice_im.append(im)
            plt.close(fig)
        
        img_grid = make_grid(slice_im, 5)
        sdf3d[ic] = ToPILImage()(img_grid)
    
    return {'sdf3d' : sdf3d}

def draw_unet3d_result(img, pred, boundary, ls=None, err=None, num_slice=10, alpha=0.2, levels=[-0.5, -0.1, 0, 0.1, 0.5], err_rng=[0, 0.1]):
    nc, nx, ny, nz = boundary.shape

    pred3d = {}
    sdf3d = {}
    err3d = {}
    for ic in range(1, nc):
        slice_idx = select_slice(boundary[[ic]], num_slice = num_slice)
        slice_pi = []
        slice_pb = []
        for s in slice_idx:
            simg = img[[0],:,:, s]
            sbimg = boundary[[ic], :, :, s]

            simg = (simg * 255).repeat(3,1,1).to(torch.uint8)
            sbimg = (sbimg * 255).repeat(3,1,1).to(torch.uint8)
            mask = pred[[ic],:,:,s].to(torch.bool)

            pred_i = draw_segmentation_masks(simg, mask, colors=COLORS[ic], alpha=alpha)
            pred_b = draw_segmentation_masks(sbimg, mask, colors=COLORS[ic], alpha=alpha)

            slice_pi.append(pred_i)
            slice_pb.append(pred_b)
        
        slice_pi = make_grid(slice_pi, 5)
        slice_pb = make_grid(slice_pb, 5)

        pred3d[ic] = (slice_pi, slice_pb) 
        
        if ls is not None:
            slice_bl = []
            for s in slice_idx:
                sb = boundary[ic, :, :, s]
                sl = ls[ic, :, :, s]

                x, y = np.linspace(-1,1,num=nx), np.linspace(-1,1,num=ny)
                X, Y = np.meshgrid(x, y)

                fig = Figure(figsize=(4,4), dpi=100)
                fig.subplots_adjust(0,0,1,1)
                canvas = FigureCanvasAgg(fig)
                ax = fig.add_subplot()
                ax.imshow(sb, cmap='gray', extent=(-1,1,1,-1))
                CS = ax.contour(X, Y, sl, levels=levels, colors=COLORS_[ic])
                ax.clabel(CS, inline=True)
                ax.axis('off')
                canvas.draw()
                im = Image.fromarray(np.asarray(canvas.buffer_rgba()))
                im = PILToTensor()(im)
                slice_bl.append(im)
                plt.close(fig)
            
            slice_bl = make_grid(slice_bl, 5)
            sdf3d[ic] = slice_bl
        
        if err is not None:
            slice_err = []
            for s in slice_idx:
                se = err[ic, :, :, s]

                fig = Figure(figsize=(4,4), dpi=100)
                canvas = FigureCanvasAgg(fig)
                ax = fig.add_subplot()
                im = ax.imshow(se, cmap='jet', extent=(-1,1,1,-1), vmin=err_rng[0], vmax=err_rng[1])
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='2%', pad=0.1)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                cbar.formatter.set_powerlimits((0, 0))
                cbar.formatter.set_useMathText(True)
                canvas.draw()
                im = Image.fromarray(np.asarray(canvas.buffer_rgba()))
                im = PILToTensor()(im)
                plt.close(fig)
                slice_err.append(im)
            slice_err = make_grid(slice_err, 5)
            err3d[ic] = slice_err

    vis_out = {'pred':pred3d}

    if ls is not None:
        vis_out['sdf'] = sdf3d
    
    if err is not None:
        vis_out['err'] = err3d    
    
    return vis_out


def random_subvolume(u, grids, sub_dim, spatial=3):

    if spatial == 3:
        nx, ny, nz = u.shape[2:]
        nx_sub, ny_sub, nz_sub = sub_dim

        xs = torch.tensor(np.random.choice(nx, nx_sub)).to(u).int()
        ys = torch.tensor(np.random.choice(ny, ny_sub)).to(u).int()
        zs = torch.tensor(np.random.choice(nz, nz_sub)).to(u).int()

        u_sub = torch.index_select(u, 2, xs)
        u_sub = torch.index_select(u_sub, 3, ys)
        u_sub = torch.index_select(u_sub, 4, zs)

        grids_sub = torch.index_select(grids, 2, xs)
        grids_sub = torch.index_select(grids_sub, 3, ys)
        grids_sub = torch.index_select(grids_sub, 4, zs)

    if spatial == 2:
        nx, ny = u.shape[2:]
        nx_sub, ny_sub = sub_dim

        xs = torch.tensor(np.random.choice(nx, nx_sub)).to(u).int()
        ys = torch.tensor(np.random.choice(ny, ny_sub)).to(u).int()

        u_sub = torch.index_select(u, 2, xs)
        u_sub = torch.index_select(u_sub, 3, ys)

        grids_sub = torch.index_select(grids, 2, xs)
        grids_sub = torch.index_select(grids_sub, 3, ys)

    return u_sub, grids_sub