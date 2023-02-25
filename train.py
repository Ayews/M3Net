import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
#from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from eval_during_train import get_pred_dir#,eval
import os
# IoU Loss
def iou_loss(pred, mask):
    pred  = torch.sigmoid(pred)
    inter = (pred*mask).sum(dim=(2,3))
    union = (pred+mask).sum(dim=(2,3))
    iou  = 1-(inter+1)/(union-inter+1)
    return iou.mean()

def structure_loss(pred, mask, weight=None):
    def generate_smoothed_gt(gts):
        epsilon = 0.001
        new_gts = (1-epsilon)*gts+epsilon/2
        return new_gts
    if weight == None:
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    else:
        weit = 1 + 5 * weight

    new_gts = generate_smoothed_gt(mask)
    wbce = F.binary_cross_entropy_with_logits(pred, new_gts, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def wbce(pred,mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=15, stride=1, padding=7) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    return wbce.mean()

def train_one_epoch(epoch,epochs,model,opt,train_dl):
    epoch_total_loss = 0
    epoch_loss1 = 0
    epoch_loss2 = 0
    epoch_loss3 = 0
    epoch_loss4 = 0

    loss_weights = [1, 0.8, 0.5, 0.5, 0.5]
    l = 0

    progress_bar = tqdm(train_dl, desc='Epoch[{:03d}/{:03d}]'.format(epoch+1, epochs),ncols=140)
    for i, data_batch in enumerate(progress_bar):

        l = l+1

        images, label_224, label_1_16, label_1_8, label_1_4 = data_batch

        images, label_224, = images.cuda(non_blocking=True), label_224.cuda(non_blocking=True)

        label_1_16, label_1_8, label_1_4 = label_1_16.cuda(), label_1_8.cuda(), label_1_4.cuda()

        out2, out3, out4, out5 = model(images)
        
        loss4  = F.binary_cross_entropy_with_logits(out2, label_1_16) + iou_loss(out2, label_1_16)
        loss3  = F.binary_cross_entropy_with_logits(out3, label_1_8) + iou_loss(out3, label_1_8)
        loss2  = F.binary_cross_entropy_with_logits(out4, label_1_4) + iou_loss(out4, label_1_4)
        loss1  = F.binary_cross_entropy_with_logits(out5, label_224) + iou_loss(out5, label_224)

        loss = loss_weights[0] * loss1 + loss_weights[1] * loss2 + loss_weights[2] * loss3 + loss_weights[3] * loss4

        opt.zero_grad()
        loss.backward()
        opt.step()

        epoch_total_loss += loss.cpu().data.item()
        epoch_loss1 += loss1.cpu().data.item()
        epoch_loss2 += loss2.cpu().data.item()
        epoch_loss3 += loss3.cpu().data.item()
        epoch_loss4 += loss4.cpu().data.item()

        progress_bar.set_postfix(loss=f'{epoch_loss1/(i+1):.3f}')
    return epoch_loss1/l
        
def fit(epochs, model, lr, train_dl, method):
    #progress_bar = tqdm(range(epochs))#,desc='Epoch[{:03d}/{:03d}]'.format(epochs, 40))
    #f = 'loss.txt'

    opt = get_opt(lr,model)

    for epoch in range(epochs):

        model.train()
        loss = train_one_epoch(epoch,epochs,model,opt,train_dl)

        #with open(f, 'a') as fe:
            #fe.write(str(sum(epochs[:st])+epoch+1)+'\t{loss:.3f}\n'.format(loss = loss))
                            # fe.write(str(sum(epochs[:st])+epoch+1)+'\t{loss1:.3f}\t{loss2:.3f}\t{loss3:.3f}\t{loss4:.3f}\t{lossp:.3f}\n'.format(loss1 = loss[0],loss2 = loss[1],loss3 = loss[2],loss4 = loss[3],lossp = loss[4]))

        #writer.add_scalar('Loss/train', loss, sum(epochs[:st])+epoch+1)

        '''
        if epoch % 20 == 19:
            torch.save(model.state_dict(),"savepth/tmp/"+str(sum(epochs[:st])+epoch+1))
            get_pred_dir(model)
            thread = threading.Thread(target = eval)
            thread.start()
        '''

        #torch.save(model.state_dict(),"save/tmp/"+str(sum(epochs[:st+1]))+'.pth')
    #torch.save(model.state_dict(),"save/"+method+str(sum(epochs[:step+1]))+'.pth')

def get_opt(lr,model):
    
    base_params = [params for name, params in model.named_parameters() if ("encoder" in name)]
    other_params = [params for name, params in model.named_parameters() if ("encoder" not in name)]
    params = [{'params': base_params, 'lr': lr*0.1},
          {'params': other_params, 'lr': lr}
         ]
         
    opt = torch.optim.Adam(params, lr)

    return opt

def eval():
    os.system('python eval/eval.py --method-json eval/examples/config_method_json_during_train.json --dataset-json eval/examples/config_dataset_json_during_train.json --record-txt results/r0.txt')