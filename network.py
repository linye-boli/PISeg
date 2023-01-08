from monai.networks.nets import UNet as UNet_monai
from monai.networks.nets import VNet as VNet_monai 
from monai.networks.layers import Norm 

from torchvision.models.feature_extraction import create_feature_extractor
import torch.nn.functional as F
import torch 
import torch.nn as nn
from monai.networks.blocks import Convolution
from monai.networks.layers import MedianFilter
import numpy as np
from utils import coords_like, gradient_norm

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super(MLP, self).__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ELU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)

class UNet(nn.Module):
    def __init__(
        self, 
        in_channels,
        out_channels,
        spatial_dims=3, 
        tanh=False):
        super(UNet, self).__init__()

        if spatial_dims == 3:
            unet = UNet_monai(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                channels=(32, 64, 128, 256, 512),
                strides=(2, 2, 2, 2),
                num_res_units=2,
                norm=Norm.BATCH,)
        else:
            unet = UNet_monai(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                channels=(32, 64, 128, 256, 512),
                strides=(2, 2, 2, 2),
                num_res_units=2,
                norm=Norm.BATCH,)

        down, skip, up = list(unet.model.children())
        self.backbone = nn.Sequential(*[down, skip])
        if tanh:
            self.tanh = nn.Tanh()
        else:
            self.tanh = None

        self.header = up

    def forward(self, img):
        feats = self.backbone(img)
        outputs = self.header(feats)
        
        if self.tanh is not None:
            outputs = self.tanh(outputs)

        return {'outputs' : outputs}

class PINNS(nn.Module):
    def __init__(
        self,
        spatial_dim=3,
        out_channels=2,
        num_rnd=100000):
        super(PINNS, self).__init__()
        self.mlp = MLP(in_dim=spatial_dim, out_dim=out_channels, hidden_list=[32] * 4)
        self.spatial_dim=spatial_dim
        self.out_channels=out_channels
        self.num_rnd = num_rnd
    
    def forward(self, x):
        if self.spatial_dim == 3:
            b, nc, nx, ny, nz = x.shape 
            y_grids = x.permute(0,2,3,4,1).reshape(-1,3)
            u_grids = self.mlp(y_grids).reshape(b, nx, ny, nz, self.out_channels).permute(0,4,1,2,3)
            
            y_rnd = (torch.rand((self.num_rnd, 3)).to(x) - 0.5) * 2
            y_rnd.requires_grad_()
            u_rnd = self.mlp(y_rnd)
            gnorm = gradient_norm(y_rnd, u_rnd)   

        if self.spatial_dim == 2:
            b, nc, nx, ny = x.shape 
            y_grids = x.permute(0,2,3,1).reshape(-1,2)
            u_grids = self.mlp(y_grids).reshape(b, nx, ny, self.out_channels).permute(0,3,1,2)
            
            y_rnd = (torch.rand((self.num_rnd, 2)).to(x) - 0.5) * 2
            y_rnd.requires_grad_()
            u_rnd = self.mlp(y_rnd)
            gnorm = gradient_norm(y_rnd, u_rnd)                

        return {'outputs':u_grids, 'grad_norms':gnorm}

class DeepONet(nn.Module):
    def __init__(self, spatial_dim=3, out_channels=2, feat_dim=200, num_rnd=100000):
        super(DeepONet, self).__init__()

        self.out_channels = out_channels
        self.spatial_dim = spatial_dim
        branches = []
        for i in range(out_channels):
            branches.append(['branch_{:}'.format(i), MLP(in_dim=feat_dim, out_dim=64, hidden_list=[64] * 4)])
        self.branches = nn.ModuleDict(branches)
        self.trunk = MLP(in_dim=spatial_dim, out_dim=64, hidden_list=[64] * 4)
        self.num_rnd = num_rnd
    
    def branch_forward(self, f, t):
        u = []
        for i in range(self.out_channels):
            b = self.branches['branch_{:}'.format(i)](f)
            u.append((b*t).sum(axis=1, keepdim=True))
        u = torch.cat(u, axis=1)
        return u

    def forward(self, x, feats):
        if self.spatial_dim == 3:
            b, nc, nx, ny, nz = feats.shape 
            y_grids = x.permute(0,2,3,4,1).reshape(-1,3)
            f_grids = feats.permute(0,2,3,4,1).reshape(-1,nc)
            t_grids = self.trunk(y_grids)
            u_grids = self.branch_forward(f_grids, t_grids)
            u_grids = u_grids.reshape(b,nx,ny,nz,self.out_channels).permute(0,4,1,2,3)
            
            y_rnd, f_rnd = sample_feats(feats, num_sample=self.num_rnd)
            y_rnd.requires_grad_()
            t_rnd = self.trunk(y_rnd)
            u_rnd = self.branch_forward(f_rnd, t_rnd)
            gnorm = gradient_norm(y_rnd, u_rnd)   

        if self.spatial_dim == 2:
            b, nc, nx, ny = feats.shape 
            y_grids = x.permute(0,2,3,1).reshape(-1,2)
            f_grids = feats.permute(0,2,3,1).reshape(-1,nc)
            t_grids = self.trunk(y_grids)

            u_grids = self.branch_forward(f_grids, t_grids)
            u_grids = u_grids.reshape(b,nx,ny,self.out_channels).permute(0,3,1,2)

            if self.training:
                y_rnd, f_rnd = sample_feats(feats, num_sample=self.num_rnd, spatial_dim=2)
            else:
                y_rnd, f_rnd = sample_feats(feats, num_sample=self.num_rnd, spatial_dim=2, is_grid=True)

            y_rnd.requires_grad_()
            t_rnd = self.trunk(y_rnd)
            u_rnd = self.branch_forward(f_rnd, t_rnd)
            gnorm = gradient_norm(y_rnd, u_rnd)      
            if not self.training:
                gnorm = gnorm.reshape(b,self.out_channels,nx,ny)

        return {'outputs':u_grids, 'grad_norms':gnorm}


from utils import sample_feats
    
class CNNDeepONet(nn.Module):
    def __init__(self, out_channels, spatial_dims=3, feat_dim=32):
        super(CNNDeepONet, self).__init__()
        unet = UNet_monai(
            spatial_dims=spatial_dims,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,)
        
        return_nodes = {
            'model.1.submodule.1.submodule.1.submodule.1.cat':'feat'}
        
        self.backbone = create_feature_extractor(unet, return_nodes=return_nodes)
        self.out_channels = out_channels
        self.spatial_dims = spatial_dims
        
        self.bottleneck = Convolution(spatial_dims=spatial_dims, in_channels=384, out_channels=feat_dim, kernel_size=1, strides=1, padding=0, adn_ordering='A')
        self.deeponet = DeepONet(spatial_dim=spatial_dims, out_channels=out_channels, feat_dim=feat_dim)

    def forward(self, x):
        grids = coords_like(x, permute=True, spatial=self.spatial_dims)
        f = self.backbone(x)['feat']
        f = self.bottleneck(f)

        if self.spatial_dims == 3:
            nb, nc, nx, ny, nz = x.shape
            f = f.mean(axis=(2,3,4), keepdim=True)
            feats = f.repeat(1,1,nx,ny,nz)
        elif self.spatial_dims == 2:
            nb, nc, nx, ny = x.shape
            f = f.mean(axis=(2,3), keepdim=True)
            feats = f.repeat(1,1,nx,ny)

        outputs = self.deeponet(grids, feats)

        return outputs

from monai.networks.nets import EfficientNetBNFeatures
class FADeepONet(nn.Module):
    def __init__(self, out_channels, spatial_dims=3, feat_dim=16):
        super(FADeepONet, self).__init__()
        if spatial_dims == 3:
            unet = UNet_monai(
                spatial_dims=spatial_dims,
                in_channels=4,
                out_channels=out_channels,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
                norm=Norm.BATCH,)
        else:
            backbone = EfficientNetBNFeatures('efficientnet-b0', spatial_dims=2, in_channels=1)

        self.out_channels = out_channels
        self.spatial_dims = spatial_dims
        
        self.backbone = backbone

        if out_channels == 3:
            self.enc1 = Convolution(spatial_dims=spatial_dims, in_channels=32, out_channels=feat_dim, kernel_size=5, strides=1, padding=1, adn_ordering='A')
            self.enc2 = Convolution(spatial_dims=spatial_dims, in_channels=64, out_channels=feat_dim, kernel_size=3, strides=1, padding=1, adn_ordering='A')
            self.enc3 = Convolution(spatial_dims=spatial_dims, in_channels=128, out_channels=feat_dim, kernel_size=3, strides=1, padding=1, adn_ordering='A')
            self.enc4 = Convolution(spatial_dims=spatial_dims, in_channels=384, out_channels=feat_dim*2, kernel_size=1, strides=1, padding=0, adn_ordering='A')
        elif out_channels == 2:
            self.enc1 = Convolution(spatial_dims=spatial_dims, in_channels=16+feat_dim*3, out_channels=feat_dim*4, kernel_size=5, strides=1, padding=0, adn_ordering='A')
            self.enc2 = Convolution(spatial_dims=spatial_dims, in_channels=24+feat_dim*2, out_channels=feat_dim*3, kernel_size=4, strides=1, padding=0, adn_ordering='A')
            self.enc3 = Convolution(spatial_dims=spatial_dims, in_channels=40+feat_dim, out_channels=feat_dim*2, kernel_size=3, strides=1, padding=0, adn_ordering='A')
            self.enc4 = Convolution(spatial_dims=spatial_dims, in_channels=112, out_channels=feat_dim, kernel_size=2, strides=1, padding=0, adn_ordering='A')
            self.enc5 = Convolution(spatial_dims=spatial_dims, in_channels=320, out_channels=feat_dim, kernel_size=1, strides=1, padding=0, adn_ordering='A')
            # self.enc = Convolution(
            #     spatial_dims=spatial_dims, in_channels=feat_dim * 5, out_channels=feat_dim * 5, 
            #     kernel_size=3, strides=1, padding=1, act='tanh', adn_ordering='A')

            self.avg = nn.AvgPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')
            
        self.deeponet = DeepONet(spatial_dim=spatial_dims, out_channels=out_channels, feat_dim=feat_dim * 5)

    def forward(self, x):
        grids = coords_like(x, permute=True, spatial=self.spatial_dims)
        
        feats = self.backbone(x)
        f1, f2, f3, f4, f5 = feats

        # import pdb 
        # pdb.set_trace()

        f = self.enc4(self.up(f4))
        
        f3_coords = coords_like(f3, permute=True, spatial=self.spatial_dims)
        f = F.grid_sample(f, f3_coords, mode='bilinear', align_corners=True)
        f = self.enc3(self.up(torch.cat([f, f3], axis=1)))

        f2_coords = coords_like(f2, permute=True, spatial=self.spatial_dims)
        f = F.grid_sample(f, f2_coords, mode='bilinear', align_corners=True)
        f = self.enc2(self.up(torch.cat([f, f2], axis=1)))

        f1_coords = coords_like(f1, permute=True, spatial=self.spatial_dims)
        f = F.grid_sample(f, f1_coords, mode='bilinear', align_corners=True)
        f = self.enc1(self.up(torch.cat([f, f1], axis=1)))
        
        # f1 = self.enc1(self.up1(f1))
        # f2 = self.enc2(self.up2(f2))
        # f3 = self.enc3(self.up3(f3))
        
        f5 = self.enc5(f5)
        
        # import pdb 
        # pdb.set_trace()
        # f1 = F.grid_sample(f1, grids, mode='bilinear', align_corners=True)
        # f2 = F.grid_sample(f2, grids, mode='bilinear', align_corners=True)
        # f3 = F.grid_sample(f3, grids, mode='bilinear', align_corners=True)
        # f4 = F.grid_sample(f4, grids, mode='bilinear', align_corners=True)

        if self.spatial_dims == 3:
            nb, nc, nx, ny, nz = x.shape
            fg = f4.mean(axis=(2,3,4), keepdim=True)
            fg = fg.repeat(1,1,nx,ny,nz)
        elif self.spatial_dims == 2:
            nb, nc, nx, ny = x.shape
            fg = f5.mean(axis=(2,3), keepdim=True)
            fg = fg.repeat(1,1,nx,ny)
        
        f = F.grid_sample(f, grids, mode='bilinear', align_corners=True)
        feats = torch.cat([f, fg], axis=1)
        feats = F.grid_sample(feats, grids, mode='bilinear', align_corners=True)
        # import pdb 
        # pdb.set_trace()

        outputs = self.deeponet(grids, feats)

        return outputs

from pino.fourier2d import FNO2d
from pino.fourier3d import FNO3d

class FNO(nn.Module):
    def __init__(self, in_channels, out_channels, spatial_dims=3):
        super(FNO, self).__init__()

        self.in_channels = in_channels        
        self.out_channels = out_channels
        self.spatial_dims = spatial_dims

        if spatial_dims == 2:
            self.fno2d = FNO2d(
                modes1=[12]*4, modes2=[12]*4, in_dim=in_channels + spatial_dims, out_dim=out_channels, width=64)
        if spatial_dims == 3:
            self.fno3d = FNO3d(
                modes1=[12]*4, modes2=[12]*4, modes3=[6]*4,
                in_dim=in_channels + spatial_dims, out_dim=out_channels, width=32)
        

    def forward(self, img):

        if self.spatial_dims == 2:
            grids = coords_like(img, spatial=2, permute=False)
            x = torch.cat([img, grids], axis=1).permute(0,2,3,1)        
            u = self.fno2d(x)
            u_grids = u.permute(0,3,1,2)
        if self.spatial_dims == 3:
            grids = coords_like(img, spatial=3, permute=False)
            x = torch.cat([img, grids], axis=1).permute(0,2,3,4,1)        
            u = self.fno3d(x)
            u_grids = u.permute(0,4,1,2,3)

        return {'outputs':u_grids}

from uno.uno3d import UNO3D
from uno.uno2d import UNO2D

class UNO(nn.Module):
    def __init__(self, in_channels, out_channels, spatial_dims=3):
        super(UNO, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_dims = spatial_dims

        if spatial_dims == 2:
            self.uno2d = UNO2D(in_width=spatial_dims+in_channels, width=32, out_channel=out_channels)
        if spatial_dims == 3:
            self.uno3d = UNO3D(in_width=spatial_dims+in_channels, width=16, out_channel=out_channels)
    
    def forward(self, img):
        if self.spatial_dims == 2:
            grids = coords_like(img, spatial=2, permute=False)
            x = torch.cat([img, grids], axis=1).permute(0,2,3,1)        
            u = self.uno2d(x)
            u_grids = u.permute(0,3,1,2)
        if self.spatial_dims == 3:
            grids = coords_like(img, spatial=3, permute=False)
            x = torch.cat([img, grids], axis=1).permute(0,2,3,4,1)        
            u = self.uno3d(x)
            u_grids = u.permute(0,4,1,2,3)

        return {'outputs':u_grids}




if __name__ == '__main__':
    img3d = torch.rand((4,1,64,64,32))
    img2d = torch.rand((4,1,64,64))
    feat3d = torch.rand((4,256,64,64,32))
    feat2d = torch.rand((4,256,64,64))
    grids3d = coords_like(img3d, spatial=3)
    grids2d = coords_like(img2d, spatial=2)

    # deeponet3d = DeepONet(spatial_dim=3, out_channels=2, feat_dim=256)
    # outputs = deeponet3d(grids3d, feat3d)

    deeponet2d = DeepONet(spatial_dim=2, out_channels=5, feat_dim=256)
    u2d = deeponet2d(grids2d, feat2d)