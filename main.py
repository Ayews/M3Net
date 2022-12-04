from random import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import *

from torch.utils.tensorboard import SummaryWriter
import argparse
from torch.utils import data
#from Models.t2t_vit import T2t_vit_t_14
from train import *
from Models.swin import *
from multiscale_fusion_sod import SOD

#from ImageDepthNet import ImageDepthNet
parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser()
# ICON-V: VGG16, ICON-R: ResNet50, ICON-S: Swin, ICON-P: PVTv2, CycleMLP: B4
parser.add_argument("--model", default='ICON-C')
# DUTS for Saliency, COD10K for Camouflaged, SOC for Attributes. 
parser.add_argument("--dataset", default='../datasets/DUTS/Train')
parser.add_argument("--lr", type=float, default=0.05)
parser.add_argument("--momen", type=float, default=0.9)  
parser.add_argument("--decay", type=float, default=1e-4)  
parser.add_argument("--batchsize", type=int, default=14)  
parser.add_argument("--epoch", type=int, default=60)  
# CPR: IoU+BCE, STR: Structure Loss, FL: F-measure loss
parser.add_argument("--loss", default='CPR')  
parser.add_argument("--savepath", default='../checkpoint/ICON/ICON-S')  
parser.add_argument("--valid", default=True)  
#train(dataset, parser)

args = parser.parse_args(args=[])

#model = ICON(args, model_name='ICON-S')
model = SOD(embed_dim=384,dim=96,img_size=224)
model.cuda()

import cv2
#writer = SummaryWriter()
#img = cv2.cvtColor(cv2.resize((cv2.imread('/mnt/disk2/dataset/MSRA10K/imgs/75.jpg')),(224,224)), cv2.COLOR_RGB2BGR)*1.0/255
#writer.add_graph(model, torch.from_numpy(np.asarray(img).astype(np.float32).transpose(2,0,1)).view(-1, 3, 224, 224).cuda())

#model.downsample.load_state_dict(torch.load('Weights/swin_small_patch4_window7_224.pth')['model'])
#model.downsample.load_state_dict(torch.load('Weights/swin_s_conformer_reload.pth'))
#model.load_state_dict(torch.load('save/swin_s_conformer_concat_conv.pth'))
#dict = model.rgb_backbone.state_dict()
#dict.update(torch.load('Weights/swin_base_patch4_window7_224.pth')['model'])
model.encoder.load_state_dict(torch.load('/home/yy/pretrained_model/swin_small_patch4_window7_224.pth')['model'])
#model.load_state_dict(torch.load('save/swin_b_vst_Adam_140.pth'))
#model.load_state_dict(torch.load('save/swin_t_vst80.pth'))
#model.load_state_dict(torch.load('save/multiscale_fusion_sod_d2_int2_cpr100.pth'))
model.train()
train_dataset = get_loader('DUTS/DUTS-TR', "/home/yy/datasets/", 224, mode='train')
train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle = True, 
                                               pin_memory=True,num_workers = 4
                                               )
#train_dataset2 = get_loader('MSRA10K', "/mnt/disk2/dataset/", 224, mode='train')
#train_dl2 = torch.utils.data.DataLoader(train_dataset2, batch_size=10, shuffle = True, 
#                                               pin_memory=True,
#                                               )

method = 'multiscale_fusion_sod_d2_int4d_norm_cpr'
#method = 'icon'
#f = 'lossA2.txt'
#step 1
lr = 0.0001
fit([100],model,lr,train_dl,method)
torch.save(model.state_dict(), 'save/'+method+'100.pth')
lr = 0.00002
fit([20],model,lr,train_dl,method)
torch.save(model.state_dict(), 'save/'+method+'120.pth')


#writer.close()
'''
#step 2
writer = SummaryWriter()
lr = 0.0001
fit([20,10,10],model,lr,train_dl,method,writer)
torch.save(model.state_dict(), 'save/'+method+'.pth')
writer = SummaryWriter()
'''