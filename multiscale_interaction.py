import torch
import torch.nn as nn
from timm.models.layers import DropPath
from Models.modules import MutualSelfBlock,MutualAttention
class MultiscaleInteractionBlock(nn.Module):
    def __init__(self,dim,dim1,dim2=None,embed_dim = 384,drop_path = 0.):
        super(MultiscaleInteractionBlock, self).__init__()
        self.ia1 = MutualSelfBlock(dim1=dim,dim2=dim1,dim=embed_dim,num_heads=6)
        self.dim2 = dim2
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        if self.dim2:
            self.ia2 = MutualSelfBlock(dim1=dim,dim2=dim2,dim=embed_dim,num_heads=6)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self,x1,x2,x3=None):
        #x1 = self.norm1(x1)
        #x2 = self.norm2(x2)
        x11 = self.ia1(x1,x2)
        if x3 != None:
            #x3 = self.norm3(x3)
            x12 = self.ia2(x1,x3)
        x1 = x1+x11
        if self.dim2:
            x1 = x1+x12
        x = x1 + self.drop_path(self.mlp(self.norm(x1)))
        return x

        return fea
        '''
        if self.dim3:
            return x,x0,x11,x12
        else:
            return x,x0,x11
        '''

if __name__ == '__main__':
    model = MultiscaleInteractionBlock(dim1=96,dim2=192,dim3=384)
    model.cuda()
    f = []
    f.append(torch.randn((1,3136,96)).cuda())
    f.append(torch.randn((1,784,192)).cuda())
    f.append(torch.randn((1,196,384)).cuda())
    y = model(f[0],f[1],f[2])
    print(y.shape)

