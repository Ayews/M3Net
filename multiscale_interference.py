import torch
import torch.nn as nn
from timm.models.layers import DropPath
from Models.modules import MutualSelfBlock
class multiscale_cross_attention(nn.Module):
    def __init__(self):
        super(multiscale_cross_attention, self).__init__()
        self.intef1 = MutualSelfBlock(in_dim=192,dim=384,num_heads=6)
        self.intef2 = MutualSelfBlock(in_dim=96,dim=192,num_heads=6)

    def forward(self,f):
        x3,x2,x1,_= f #x1:1/16
        x2_ = x2 + self.intef1(x2,x1)
        x3_ = x3 + self.intef2(x3,x2)
        return x1,x2_,x3_




if __name__ == '__main__':
    model = multiscale_cross_attention()
    model.cuda()
    f = []
    f.append(torch.randn((1,3136,96)).cuda())
    f.append(torch.randn((1,784,192)).cuda())
    f.append(torch.randn((1,196,384)).cuda())
    y = model(f)
    print(y[0].shape)
    print(y[1].shape)
    print(y[2].shape)
