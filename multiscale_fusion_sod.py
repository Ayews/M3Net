import torch
import torch.nn as nn
from Models.swin import SwinTransformer
from multiscale_fusion import decoder
from multiscale_interference import multiscale_cross_attention
from multiscale_interaction import MultiscaleInteractionBlock
class SOD(nn.Module):
    def __init__(self,embed_dim=384,dim=96,img_size=224):
        super(SOD, self).__init__()
        self.img_size = img_size
        self.encoder = SwinTransformer(img_size=224, 
                                           embed_dim=96,
                                           depths=[2,2,18,2],
                                           num_heads=[3,6,12,24],
                                           window_size=7)
        self.interact1 = MultiscaleInteractionBlock(dim1=384,dim2=768,embed_dim=embed_dim)
        self.interact2 = MultiscaleInteractionBlock(dim1=192,dim2=384,dim3=768,embed_dim=embed_dim)
        self.interact3 = MultiscaleInteractionBlock(dim1=96,dim2=192,dim3=384,embed_dim=embed_dim)
        
        self.decoder = decoder(embed_dim=embed_dim,dim=dim,img_size=img_size)



    def forward(self,x):
        x1,x2,x3,x4 = self.encoder(x)
        x3_ = self.interact1(x3,x4)
        x2_ = self.interact2(x2,x3_,x4)
        x1_ = self.interact3(x1,x2_,x3_)
        x = self.decoder([x3_,x2_,x1_])
        return x

if __name__ == '__main__':
    model = SOD(embed_dim=384,dim=96,img_size=224)
    model.cuda()
    
    f = torch.randn((1,3,224,224))
    x = model(f.cuda())
    print(x[3].shape)
    s1 = sum([param.nelement() for param in model.encoder.parameters()])
    s21 = sum([param.nelement() for param in model.interact1.parameters()])
    s22 = sum([param.nelement() for param in model.interact2.parameters()])
    s23 = sum([param.nelement() for param in model.interact3.parameters()])
    s3 = sum([param.nelement() for param in model.decoder.parameters()])
    print(s1,s21,s22,s23,s3)
    

