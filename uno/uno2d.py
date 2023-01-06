import torch 
import torch.nn as nn
import torch.nn.functional as F
from .integral_operators import OperatorBlock_2D 

class UNO2D(nn.Module):
    def __init__(self, in_width, width, out_channel):
        '''
        in_width : input channel, 3 for image_channel, x, y
        width : intermedate channel 
        '''
        super(UNO2D, self).__init__()
   
        self.in_width = in_width # input function co-domain dimention after concatenating (x,y)
        self.width = width # lifting dimension
        self.out_channel = out_channel

        self.fc = nn.Linear(self.in_width, self.width//2)
        self.fc0 = nn.Linear(self.width//2, self.width) 
        self.fc1 = nn.Linear(1*self.width, 2*self.width)
        self.fc2 = nn.Linear(2*self.width, out_channel)

        # in_codim : in_channel, 
        # out_codim : out_channel,
        # dim1 : out_x
        # dim2 : out_y
        # modes1 : fourier_compx
        # modes2 : fourier_compy
        self.G0 = OperatorBlock_2D(self.width, 2*self.width, 32, 32, 14, 14)
        self.G1 = OperatorBlock_2D(2*self.width, 4*self.width, 16, 16, 6,6)
        self.G2 = OperatorBlock_2D(4*self.width, 8*self.width, 8, 8,3,3)
        self.G3 = OperatorBlock_2D(8*self.width, 16*self.width, 4, 4,2,2)
        self.G4 = OperatorBlock_2D(16*self.width, 16*self.width, 4, 4,2,2)
        self.G5 = OperatorBlock_2D(16*self.width, 16*self.width, 4, 4,2,2)
        self.G6 = OperatorBlock_2D(16*self.width, 16*self.width, 4, 4,2,2)        
        self.G7 = OperatorBlock_2D(16*self.width, 16*self.width, 4, 4,2,2)
        self.G8 = OperatorBlock_2D(16*self.width, 16*self.width, 4, 4,2,2)   
        self.G9 = OperatorBlock_2D(16*self.width, 8*self.width, 8, 8,2,2)
        self.G10 = OperatorBlock_2D(16*self.width, 4*self.width, 16, 16,3,3)
        self.G11 = OperatorBlock_2D(8*self.width, 2*self.width, 32, 32,6,6)
        self.G12 = OperatorBlock_2D(4*self.width, self.width, 64, 64,14,14) # will be reshaped
    
    def forward(self, x):
        # x : bxyc
        x_fc = self.fc(x) # c: 3 -> 8
        x_fc = F.gelu(x_fc)

        x_fc0 = self.fc0(x_fc) # c : 8 -> 16 (width)
        x_fc0 = F.gelu(x_fc0) 

        x_fc0 = x_fc0.permute(0, 3, 1, 2)

        D1,D2 = x_fc0.shape[-2], x_fc0.shape[-1] # spatial dim, nx : 64, ny : 64

        x_c0 = self.G0(x_fc0,D1//2,D2//2)  # 32x32x32
        x_c1 = self.G1(x_c0,D1//4,D2//4)   # 64x16x16
        x_c2 = self.G2(x_c1,D1//8,D2//8)   # 128x8x8
        x_c3 = self.G3(x_c2,D1//16,D2//16) # 256x4x4
        x_c4 = self.G4(x_c3,D1//16,D2//16) # 256x4x4
        x_c5 = self.G5(x_c4,D1//16,D2//16) # 256x4x4
        x_c6 = self.G6(x_c5,D1//16,D2//16) # 256x4x4
        x_c7 = self.G7(x_c6,D1//16,D2//16) # 256x4x4
        x_c8 = self.G8(x_c7,D1//16,D2//16) # 256x4x4
        x_c9 = self.G9(x_c8,D1//8,D2//8)   # 128x8x8
        x_c9 = torch.cat([x_c9, x_c2], dim=1) # (128+128)x8x8
        x_c10 = self.G10(x_c9 ,D1//4,D2//4) # 64x16x16
        x_c10 = torch.cat([x_c10, x_c1], dim=1) # (64+64)x16x16
        x_c11 = self.G11(x_c10 ,D1//2,D2//2)  # 32x32x32
        x_c11 = torch.cat([x_c11, x_c0], dim=1) # (32+32)x32x32
        x_c12 = self.G12(x_c11, D1, D2) # 16xD1_outxD2_out

        x_c12 = x_c12.permute(0, 2, 3, 1)
        x_fc1 = self.fc1(x_c12)
        x_fc1 = F.gelu(x_fc1)
        x_out = self.fc2(x_fc1)
        
        return x_out