from .integral_operators import OperatorBlock_3D
import torch.nn as nn
import torch.nn.functional as F
import torch

class UNO3D(nn.Module):
    def __init__(self, in_width, width, out_channel):
        super(UNO3D, self).__init__()
        self.in_width = in_width
        self.width = width 
        self.out_channel = out_channel
        self.fc = nn.Linear(self.in_width, self.width//2)
        self.fc0 = nn.Linear(self.width//2, self.width) 
        self.conv0 = OperatorBlock_3D(self.width, 2*self.width, 16, 16, 8, 16, 16, 6, Normalize = False)
        self.conv1 = OperatorBlock_3D(2*self.width, 4*self.width, 8, 8, 4, 8, 8, 3, Normalize = False)
        self.conv2 = OperatorBlock_3D(4*self.width, 8*self.width, 4, 4, 2, 4, 4, 3, Normalize = False)
        self.conv3 = OperatorBlock_3D(8*self.width, 16*self.width, 4, 4, 2, 4, 4, 3, Normalize = False)
        self.conv4 = OperatorBlock_3D(16*self.width, 16*self.width, 4, 4, 2, 4, 4, 3, Normalize = False)
        self.conv5 = OperatorBlock_3D(32*self.width, 8*self.width, 4, 4, 2, 4, 4, 3, Normalize = False)
        self.conv6 = OperatorBlock_3D(16*self.width, 4*self.width, 8, 8, 4, 4, 4, 3, Normalize = False)
        self.conv7 = OperatorBlock_3D(8*self.width, 2*self.width, 16, 16, 8, 4, 4, 3, Normalize = False)
        self.conv8 = OperatorBlock_3D(4*self.width, 2*self.width, 32, 32, 8, 4, 4, 3, Normalize = False)
        self.fc1 = nn.Linear(2*self.width, 4*self.width)
        self.fc2 = nn.Linear(4*self.width, out_channel)
    
    def forward(self, x):
        x_fc_1 = self.fc(x)
        x_fc_1 = F.gelu(x_fc_1)
        x_fc0 = self.fc0(x_fc_1)
        x_fc0 = F.gelu(x_fc0)        
        x_fc0 = x_fc0.permute(0, 4, 1, 2, 3)
        D1,D2,D3 = x_fc0.shape[-3],x_fc0.shape[-2],x_fc0.shape[-1]
       
        x_c0 = self.conv0(x_fc0, D1//2, D2//2, D3)
        x_c1 = self.conv1(x_c0, D1//4, D2//4, D3//2)
        x_c2 = self.conv2(x_c1, D1//8, D2//8, D3//4)        
        x_c3 = self.conv3(x_c2, D1//16, D2//16, D3//8)
        x_c4 = self.conv4(x_c3, D1//16, D2//16, D3//8)
        x_c5 = self.conv5(torch.cat([x_c4, x_c3], axis=1), D1//8, D2//8, D3//4)
        x_c6 = self.conv6(torch.cat([x_c5, x_c2], axis=1), D1//4, D2//4, D3//2)
        x_c7 = self.conv7(torch.cat([x_c6, x_c1], axis=1), D1//2, D2//2, D3)
        x_c8 = self.conv8(torch.cat([x_c7, x_c0], axis=1), D1, D2, D3)

        x_c8 = x_c8.permute(0, 2, 3, 4, 1)
        x_fc1 = self.fc1(x_c8)
        x_fc1 = F.gelu(x_fc1)
        x_out = self.fc2(x_fc1)
        return x_out