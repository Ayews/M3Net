import torch
import torch.nn as nn
from Models.swin import SwinTransformer
from Models.icon.resnet_encoder import ResNet
from Models.t2t_vit import T2t_vit_t_14
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
        if method == 'M3Net-S':
            self.encoder = SwinTransformer(img_size=img_size, 
                                            embed_dim=dim,
                                            depths=[2,2,18,2],
                                            num_heads=[3,6,12,24],
                                            window_size=7)
        elif method == 'M3Net-R':
            self.encoder = ResNet()

        self.interact1 = MultilevelInteractionBlock(dim=dim*4,dim1=dim*8,embed_dim=embed_dim,num_heads=4,mlp_ratio=3)
        self.interact2 = MultilevelInteractionBlock(dim=dim*2,dim1=dim*4,dim2=dim*8,embed_dim=embed_dim,num_heads=2,mlp_ratio=3)
        self.interact3 = MultilevelInteractionBlock(dim=dim,dim1=dim*2,dim2=dim*4,embed_dim=embed_dim,num_heads=1,mlp_ratio=3)

        self.decoder = decoder(embed_dim=embed_dim,dim=dim,img_size=img_size,mlp_ratio=1)

    def forward(self,x):
        fea_1_4,fea_1_8,fea_1_16,fea_1_32 = self.encoder(x)

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

from thop import profile
if __name__ == '__main__':
    # Test
    model = M3Net(embed_dim=384,dim=96,img_size=224)
    model.cuda()
    
    f = torch.randn((1,3,224,224))
    x = model(f.cuda())
    #print(x[3].shape)
    
    s0 = sum([param.nelement() for param in model.parameters()])
    s1 = sum([param.nelement() for param in model.encoder.parameters()])
    s21 = sum([param.nelement() for param in model.interact1.parameters()])
    s22 = sum([param.nelement() for param in model.interact2.parameters()])
    s23 = sum([param.nelement() for param in model.interact3.parameters()])
    s3 = sum([param.nelement() for param in model.decoder.parameters()])
    print(s0,s1,s21,s22,s23,s3)
    f = torch.randn((1,3,224,224))
    f1 = torch.randn((1,56*56,96))
    f2 = torch.randn((1,28*28,192))
    f3 = torch.randn((1,14*14,384))
    #print(model.encoder.flops()/1e9)
    macs, params = profile(model, inputs=(f.cuda(),))
    print(macs/1e9,params/1e6)
    

