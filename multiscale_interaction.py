import torch
import torch.nn as nn
from timm.models.layers import DropPath
from Models.modules import MutualSelfBlock,MutualAttention
class MultiscaleInteractionBlock(nn.Module):
    def __init__(self,dim,dim1,dim2=None,embed_dim = 384,drop_path = 0.):
        super(MultiscaleInteractionBlock, self).__init__()
        self.ia1 = MutualSelfBlock(dim1=dim,dim2=dim1,dim=embed_dim,num_heads=1)
        self.dim2 = dim2
        self.norm1 = nn.LayerNorm(dim)
        self.mlp1 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        if self.dim2:
            self.ia2 = MutualSelfBlock(dim1=dim,dim2=dim2,dim=embed_dim,num_heads=1)
            self.norm2 = nn.LayerNorm(dim)
            self.mlp2 = nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )
        self.proj1 = nn.Linear(6,1)
        self.proj2 = nn.Linear(6,1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self,fea,fea_1,fea_2=None):
        att,att1 = self.ia1(fea,fea_1)
        fea = fea + att
        fea = fea + self.drop_path(self.mlp1(self.norm1(fea)))
        f1 = fea
        if fea_2 != None:
            att,att2 = self.ia2(fea,fea_2)
            fea = fea + att
            fea = fea + self.drop_path(self.mlp2(self.norm2(fea)))
        if self.dim2 == 384:
            return fea,att1,att2
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

