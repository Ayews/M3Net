from random import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import *

#from torch.utils.tensorboard import SummaryWriter
import argparse
from torch.utils import data
#from Models.t2t_vit import T2t_vit_t_14
from train import *
from Models.swin import *
from multiscale_fusion_sod import MIFSOD

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
model = MIFSOD(embed_dim=384,dim=96,img_size=224)
model.cuda()

import cv2
#writer = SummaryWriter()
#img = cv2.cvtColor(cv2.resize((cv2.imread('/mnt/disk2/dataset/MSRA10K/imgs/75.jpg')),(224,224)), cv2.COLOR_RGB2BGR)*1.0/255
#writer.add_graph(model, torch.from_numpy(np.asarray(img).astype(np.float32).transpose(2,0,1)).view(-1, 3, 224, 224).cuda())

#model.encoder.load_state_dict(torch.load('/home/yy/pretrained_model/swin_tiny_patch4_window7_224.pth')['model'])
#model.encoder.load_state_dict(torch.load('/home/yy/pretrained_model/T2T_ViTt_14.pth.tar')['state_dict_ema'])
pretrained_dict = torch.load('/home/yy/pretrained_model/resnet50.pth')
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model.encoder.state_dict()}
model.encoder.load_state_dict(pretrained_dict)
model.train()
train_dataset = get_loader('DUTS/DUTS-TR', "/home/yy/datasets/", 224, mode='train')
#train_dataset_b = get_loader('DUTS_MSRA10K', "/home/yy/datasets/", 224, mode='train')
train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle = True, 
                                               pin_memory=True,num_workers = 4
                                               )
#train_dl_b = torch.utils.data.DataLoader(train_dataset_b, batch_size=8, shuffle = True, 
#                                               pin_memory=True,num_workers = 4
#                                               )                                            
#train_dataset2 = get_loader('MSRA10K', "/mnt/disk2/dataset/", 224, mode='train')
#train_dl2 = torch.utils.data.DataLoader(train_dataset2, batch_size=10, shuffle = True, 
#                                               pin_memory=True,
#                                               )

method = 'multiscale_fusion_sod_resnet50_SET_cpr'
#method = 'icon'
#f = 'lossA2.txt'
#step 1
lr = 0.0001
fit([100],model,lr,train_dl,method)
torch.save(model.state_dict(), 'savepth/'+method+'100.pth')
lr = 0.00002
fit([20],model,lr,train_dl,method)
torch.save(model.state_dict(), 'savepth/'+method+'120.pth')
#writer.close()
