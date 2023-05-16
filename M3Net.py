import torch
import torch.nn as nn
from Models.swin import SwinTransformer
from Models.resnet import ResNet
from Models.t2t_vit import T2t_vit_t_14
from Models.EfficientNet import EfficientNet
from multistage_fusion import decoder
from multilevel_interaction import MultilevelInteractionBlock

class M3Net(nn.Module):
    r""" Multilevel, Mixed and Multistage Attention Network for Salient Object Detection. 
    
    Args:
        embed_dim (int): Dimension for attention. Default 384
        dim (int): Patch embedding dimension. Default 96
        img_size (int): Input image size. Default 224
        method (string): Backbone used as the encoder.
    """

    def __init__(self,embed_dim=384,dim=96,img_size=224,method='M3Net-S'):
        super(M3Net, self).__init__()
        self.img_size = img_size
        self.feature_dims = []
        self.method = method
        self.dim = dim
        if method == 'M3Net-S':
            self.encoder = SwinTransformer(img_size=img_size, 
                                            embed_dim=dim,
                                            depths=[2,2,18,2],
                                            num_heads=[4,8,16,32],
                                            window_size=7)
            self.interact1 = MultilevelInteractionBlock(dim=dim*4,dim1=dim*8,embed_dim=embed_dim,num_heads=4,mlp_ratio=3)
            self.interact2 = MultilevelInteractionBlock(dim=dim*2,dim1=dim*4,dim2=dim*8,embed_dim=embed_dim,num_heads=2,mlp_ratio=3)
            self.interact3 = MultilevelInteractionBlock(dim=dim,dim1=dim*2,dim2=dim*4,embed_dim=embed_dim,num_heads=1,mlp_ratio=3)
            feature_dims=[dim,dim*2,dim*4]

        elif method == 'M3Net-R':
            self.encoder = ResNet()
            self.proj1 = nn.Conv2d(256,dim,1)
            self.proj2 = nn.Conv2d(512,dim*2,1)
            self.proj3 = nn.Conv2d(1024,dim*4,1)
            self.proj4 = nn.Conv2d(2048,dim*8,1)

            self.interact1 = MultilevelInteractionBlock(dim=dim*4,dim1=dim*8,embed_dim=embed_dim,num_heads=4,mlp_ratio=3)
            self.interact2 = MultilevelInteractionBlock(dim=dim*2,dim1=dim*4,dim2=dim*8,embed_dim=embed_dim,num_heads=2,mlp_ratio=3)
            self.interact3 = MultilevelInteractionBlock(dim=dim,dim1=dim*2,dim2=dim*4,embed_dim=embed_dim,num_heads=1,mlp_ratio=3)
            feature_dims=[dim,dim*2,dim*4]


        elif method == 'M3Net-T':
            self.encoder = T2t_vit_t_14(pretrained=False)
            self.interact2 = MultilevelInteractionBlock(dim=dim,dim1=embed_dim,embed_dim=embed_dim,num_heads=2,mlp_ratio=3)
            self.interact3 = MultilevelInteractionBlock(dim=dim,dim1=dim,dim2=embed_dim,embed_dim=embed_dim,num_heads=1,mlp_ratio=3)
            feature_dims=[dim,dim,embed_dim]

        elif method == 'M3Net-E':
            self.encoder = EfficientNet.from_name(f'efficientnet-b2')
            self.proj1 = nn.Conv2d(24,dim,1)
            self.proj2 = nn.Conv2d(48,dim*2,1)
            self.proj3 = nn.Conv2d(120,dim*4,1)
            self.proj4 = nn.Conv2d(352,dim*8,1)

            self.interact1 = MultilevelInteractionBlock(dim=dim*4,dim1=dim*8,embed_dim=embed_dim,num_heads=4,mlp_ratio=3)
            self.interact2 = MultilevelInteractionBlock(dim=dim*2,dim1=dim*4,dim2=dim*8,embed_dim=embed_dim,num_heads=2,mlp_ratio=3)
            self.interact3 = MultilevelInteractionBlock(dim=dim,dim1=dim*2,dim2=dim*4,embed_dim=embed_dim,num_heads=1,mlp_ratio=3)
            feature_dims=[dim,dim*2,dim*4]
        self.decoder = decoder(embed_dim=embed_dim,dims=feature_dims,img_size=img_size,mlp_ratio=1)

    def forward(self,x):
        fea = self.encoder(x)
        if self.method == 'M3Net-S':
            fea_1_4,fea_1_8,fea_1_16,fea_1_32 = fea
            fea_1_16_ = self.interact1(fea_1_16,fea_1_32)
            fea_1_8_ = self.interact2(fea_1_8,fea_1_16_,fea_1_32)
            fea_1_4_ = self.interact3(fea_1_4,fea_1_8_,fea_1_16_)
        elif self.method == 'M3Net-R':
            fea_1_4,fea_1_8,fea_1_16,fea_1_32 = fea
            B,_,_,_ = fea_1_4.shape
            fea_1_4 = self.proj1(fea_1_4).reshape(B,self.dim,-1).transpose(1,2)
            fea_1_8 = self.proj2(fea_1_8).reshape(B,self.dim*2,-1).transpose(1,2)
            fea_1_16 = self.proj3(fea_1_16).reshape(B,self.dim*4,-1).transpose(1,2)
            fea_1_32 = self.proj4(fea_1_32).reshape(B,self.dim*8,-1).transpose(1,2)
            fea_1_16_ = self.interact1(fea_1_16,fea_1_32)
            fea_1_8_ = self.interact2(fea_1_8,fea_1_16_,fea_1_32)
            fea_1_4_ = self.interact3(fea_1_4,fea_1_8_,fea_1_16_)
        elif self.method == 'M3Net-T':
            fea_1_4,fea_1_8,fea_1_16_ = fea
            fea_1_8_ = self.interact2(fea_1_8,fea_1_16_)
            fea_1_4_ = self.interact3(fea_1_4,fea_1_8_,fea_1_16_)
        elif self.method == 'M3Net-E':
            fea_1_4,fea_1_8,fea_1_16,fea_1_32 = fea
            B,_,_,_ = fea_1_4.shape
            fea_1_4 = self.proj1(fea_1_4).reshape(B,self.dim,-1).transpose(1,2)
            fea_1_8 = self.proj2(fea_1_8).reshape(B,self.dim*2,-1).transpose(1,2)
            fea_1_16 = self.proj3(fea_1_16).reshape(B,self.dim*4,-1).transpose(1,2)
            fea_1_32 = self.proj4(fea_1_32).reshape(B,self.dim*8,-1).transpose(1,2)
            fea_1_16_ = self.interact1(fea_1_16,fea_1_32)
            fea_1_8_ = self.interact2(fea_1_8,fea_1_16_,fea_1_32)
            fea_1_4_ = self.interact3(fea_1_4,fea_1_8_,fea_1_16_)
        mask = self.decoder([fea_1_16_,fea_1_8_,fea_1_4_])
        return mask

    def flops(self):
        flops = 0
        flops += self.encoder.flops()
        N1 = self.img_size//4*self.img_size//4
        N2 = self.img_size//8*self.img_size//8
        N3 = self.img_size//16*self.img_size//16
        N4 = self.img_size//32*self.img_size//32
        flops += self.interact1.flops(N3,N4)
        flops += self.interact2.flops(N2,N3,N4)
        flops += self.interact3.flops(N1,N2,N3)
        flops += self.decoder.flops()
        return flops

#from thop import profile
if __name__ == '__main__':
    # Test
    model = M3Net(embed_dim=384,dim=64,img_size=224,method='M3Net-E')
    model.cuda()
    
    f = torch.randn((1,3,224,224))
    x = model(f.cuda())
    for m in x:
        print(m.shape)
    
    import torch
    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('macs: ', macs))
    print('{:<30}  {:<8}'.format('parameters: ', params))
    