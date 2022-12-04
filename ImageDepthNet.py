import torch.nn as nn
from Models.swin import SwinTransformer
from Models.t2t_vit import T2t_vit_t_14
from Models.Transformer import Transformer
from Models.Transformer import token_Transformer
from Models.Decoder import Decoder
import numpy as np
#from conformer_test import Conformer
#from upsample import *
def lhw(x):
    B,new_HW,C = x.shape
    x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
    return x
def hwl(x):
    B,C,H,W = x.shape
    return x.reshape(B,C,-1).transpose(1,2)
class ImageDepthNet(nn.Module):
    def __init__(self):
        super(ImageDepthNet, self).__init__()

        # VST Encoder
        self.rgb_backbone = SwinTransformer(img_size=224, 
                                           embed_dim=128,
                                           depths=[2,2,18,2],
                                           num_heads=[4,8,16,32],
                                           window_size=7)
        #self.rgb_backbone.load_state_dict(torch.load('Weights/swin_base_patch4_window7_224.pth')['model'])

        # VST Convertor
        #self.transformer = Transformer(embed_dim=384, depth=4, num_heads=6, mlp_ratio=3.)


        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(256)
        self.norm3 = nn.LayerNorm(128)
        self.mlp1 = nn.Sequential(
            nn.Linear(512, 384),
            nn.GELU(),
            nn.Linear(384,384),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64,64),

        )
        self.mlp3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64,64),
        )


        # VST Decoder
        self.token_trans = token_Transformer(embed_dim=384, depth=4, num_heads=6, mlp_ratio=3.)
        self.decoder = Decoder(embed_dim=384, token_dim=64, depth=2, img_size=224)
        #self.token_trans_c = token_Transformer(embed_dim=256, depth=4, num_heads=3, mlp_ratio=3.)
        #self.decoder_c = Decoder(embed_dim=256, token_dim=64, depth=2, img_size=224)

    def forward(self, image_Input):

        B, _, _, _ = image_Input.shape
        # VST Encoder
        f = self.rgb_backbone(image_Input)

        _,rgb_fea_1_16, rgb_fea_1_8, rgb_fea_1_4 = f[3], f[2], f[1], f[0]
        
   



        rgb_fea_1_4 = self.mlp3(self.norm3(rgb_fea_1_4))
        rgb_fea_1_8 = self.mlp2(self.norm2(rgb_fea_1_8))
        rgb_fea_1_16 = self.mlp1(self.norm1(rgb_fea_1_16))
        #rgb_fea_1_8c = self.mlp3(rgb_fea_1_8c)
        # VST Convertor
        #rgb_fea_1_16 = self.transformer(rgb_fea_1_16)
        # rgb_fea_1_16 [B, 14*14, 384]

        # VST Decoder
        saliency_fea_1_16, fea_1_16, saliency_tokens, contour_fea_1_16, contour_tokens = self.token_trans(rgb_fea_1_16)
        #saliency_fea_1_16c, fea_1_16c, saliency_tokensc, contour_fea_1_16c, contour_tokensc = self.token_trans(rgb_fea_1_16c)

        # saliency_fea_1_16 [B, 14*14, 384]
        # fea_1_16 [B, 1 + 14*14 + 1, 384]
        # saliency_tokens [B, 1, 384]
        # contour_fea_1_16 [B, 14*14, 384]
        # contour_tokens [B, 1, 384]

        outputs = self.decoder(saliency_fea_1_16, fea_1_16, saliency_tokens, contour_fea_1_16, contour_tokens, rgb_fea_1_8, rgb_fea_1_4)
        #outputsc = self.decoder(saliency_fea_1_16c, fea_1_16c, saliency_tokensc, contour_fea_1_16c, contour_tokensc, rgb_fea_1_8c, rgb_fea_1_4c)

        return outputs
