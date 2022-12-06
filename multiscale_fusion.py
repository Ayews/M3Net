import torch
import torch.nn as nn
from timm.models.layers import DropPath
from Models.modules import WindowAttentionBlock,Block,mixattentionblock,SEBlock
class decoder(nn.Module):
    def __init__(self,embed_dim,dim,img_size):
        super(decoder, self).__init__()
        self.img_size = img_size
        self.fusion1 = multiscale_fusion(in_dim=384,f_dim=192,dim=dim,kernel_size=(3,3),img_size=(img_size//8,img_size//8),stride=(2,2),padding=(1,1))
        self.fusion2 = multiscale_fusion(in_dim=192,f_dim=96,dim=dim,kernel_size=(3,3),img_size=(img_size//4,img_size//4),stride=(2,2),padding=(1,1))
        self.fusion3 = multiscale_fusion(in_dim=96,f_dim=96,dim=dim,kernel_size=(7,7),img_size=(img_size//1,img_size//1),stride=(4,4),padding=(2,2),fuse=False)

        self.mixatt1 = mixattention(in_dim=192,dim=embed_dim,img_size=(img_size//8,img_size//8),num_heads=1,mlp_ratio=3)
        self.mixatt2 = mixattention(in_dim=dim,dim=embed_dim,img_size=(img_size//4,img_size//4),num_heads=1,mlp_ratio=3)

        self.proj1 = nn.Linear(384,1)
        self.proj2 = nn.Linear(192,1)
        self.proj3 = nn.Linear(96,1)
        self.proj4 = nn.Linear(96,1)



    def forward(self,f):
        x1,x2,x3 = f #x1:1/16
        B,_,_ = x1.shape
        x2 = self.fusion1(x1,x2)
        x2 = self.mixatt1(x2)
        x3 = self.fusion2(x2,x3)
        x3 = self.mixatt2(x3)
        x4 = self.fusion3(x3)
        x1 = self.proj1(x1)
        x1 = x1.transpose(1, 2).reshape(B, 1, self.img_size // 16, self.img_size // 16)
        x2 = self.proj2(x2)
        x2 = x2.transpose(1, 2).reshape(B, 1, self.img_size // 8, self.img_size // 8)
        x3 = self.proj3(x3)
        x3 = x3.transpose(1, 2).reshape(B, 1, self.img_size // 4, self.img_size // 4)
        x4 = self.proj4(x4)
        x4 = x4.transpose(1, 2).reshape(B, 1, self.img_size // 1, self.img_size // 1)
        return [x1,x2,x3,x4]


class multiscale_fusion(nn.Module):
    def __init__(self,in_dim,f_dim,dim,kernel_size,img_size,stride,padding,fuse=True):
        super(multiscale_fusion, self).__init__()
        self.fuse = fuse
        self.norm = nn.LayerNorm(in_dim)
        self.project = nn.Linear(in_dim, in_dim * kernel_size[0] * kernel_size[1])
        self.upsample = nn.Fold(output_size=img_size, kernel_size=kernel_size, stride=stride, padding=padding)
        if self.fuse:
            self.se = SEBlock(in_dim+f_dim,16)
            self.mlp1 = nn.Sequential(
                nn.Linear(in_dim+f_dim, f_dim),
                nn.GELU(),
                nn.Linear(f_dim, f_dim),
            )
        
    def forward(self,x1,x2=None):
        x1 = self.project(self.norm(x1))
        x1 = self.upsample(x1.transpose(1,2))
        B, C, _, _ = x1.shape
        x1 = x1.view(B, C, -1).transpose(1, 2)#.contiguous()
        if self.fuse:
            x = torch.cat([x1,x2],dim=2)
            x = self.se(x)
            x = self.mlp1(x)
        else:
            x = x1
        return x

    
class mixattention(nn.Module):
    def __init__(self,in_dim,dim,img_size,num_heads=1,mlp_ratio=4,depth=2,drop_path = 0.):
        super(mixattention, self).__init__()

        self.img_size = img_size
        self.norm1 = nn.LayerNorm(in_dim)
        self.mlp1 = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.blocks = nn.ModuleList([
            mixattentionblock(dim=dim,img_size=img_size,num_heads=num_heads,mlp_ratio=mlp_ratio)
            for i in range(depth)])
        self.norm2 = nn.LayerNorm(dim)
        self.mlp2 = nn.Sequential(
            nn.Linear(dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self,x):
        x = self.mlp1(self.norm1(x))
        for blk in self.blocks:
            x = blk(x)
        x = self.drop_path(self.mlp2(self.norm2(x)))
        return x



if __name__ == '__main__':
    model = decoder(embed_dim=384,dim=96,img_size=224)
    model.cuda()
    f = []
    f.append(torch.randn((1,196,384)).cuda())
    f.append(torch.randn((1,784,192)).cuda())
    f.append(torch.randn((1,3136,96)).cuda())

    y = model(f)
    print(y[0].shape)
    print(y[1].shape)
    print(y[2].shape)
    print(y[3].shape)


