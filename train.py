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
#from apex import amp
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

def flat(mask,mask2,h1,h2):
    batch_size = mask.shape[0]

    mask = F.interpolate(mask,size=(int(h1),int(h1)), mode='bilinear')
    x = mask.view(batch_size, 1, -1).permute(0, 2, 1) 
    mask2 = F.interpolate(mask2,size=(int(h2),int(h2)), mode='bilinear')
    x2 = mask2.view(batch_size, 1, -1).permute(0, 2, 1) 
    
    # print(x.shape)  b 28*28 1
    g = x @ x2.transpose(-2,-1) # b 28*28 28*28
    g = g.unsqueeze(1) # b 1 28*28 28*28
    return g

def att_loss(pred,mask1,mask2,h1,h2):
    g = flat(mask1,mask2,h1,h2)
    #np4 = torch.sigmoid(p4.detach())
    #np5 = torch.sigmoid(p5.detach())
    #p = flat(np4,np5,h1,h2)
    #w  = torch.abs(g-p)
    #w = (w)*0.5+1
    attbce=F.binary_cross_entropy_with_logits(pred, g,reduction='mean')
    return attbce

def train_one_epoch(epoch,epochs,model,opt,train_dl):
    epoch_total_loss = 0
    epoch_loss1 = 0
    epoch_loss2 = 0
    epoch_loss3 = 0
    epoch_loss4 = 0
    epoch_lossp = 0
    epoch_loss41 = 0
    criterion = nn.BCEWithLogitsLoss()
    wc = wbce
    sl = structure_loss
    local_rank = 0
    loss_weights = [1, 0.8, 0.5, 0.5, 0.5]
    l = 0


    #opt      = torch.optim.SGD([{'params':base}, {'params':head}], lr=lr, momentum=momen, weight_decay=decay, nesterov=True)
    #model, opt = amp.initialize(model, opt, opt_level='O1') 
    progress_bar = tqdm(train_dl, desc='Epoch[{:03d}/{:03d}]'.format(epoch+1, epochs),ncols=140)
    for i, data_batch in enumerate(progress_bar):
        #opt.param_groups[0]['lr'] = (1-abs((epoch+1)/(epochs+1)*2-1))*lr*0.1
        #opt.param_groups[1]['lr'] = (1-abs((epoch+1)/(epochs+1)*2-1))*lr
        l = l+1
        #image, mask = image.cuda(), mask.cuda()  
        images, label_224, label_14, label_28, label_56, label_112, \
             = data_batch

        images, label_224, = Variable(images.cuda(local_rank, non_blocking=True)), \
                                        Variable(label_224.cuda(local_rank, non_blocking=True))

        label_14, label_28, label_56, label_112 = Variable(label_14.cuda()), Variable(label_28.cuda()),\
                                                    Variable(label_56.cuda()), Variable(label_112.cuda())

        out2, out3, out4, out5 = model(images)
        #att1,att2 = att
        
        loss4  = F.binary_cross_entropy_with_logits(out2, label_14) + iou_loss(out2, label_14)
        loss3  = F.binary_cross_entropy_with_logits(out3, label_28) + iou_loss(out3, label_28)
        loss2  = F.binary_cross_entropy_with_logits(out4, label_56) + iou_loss(out4, label_56)
        loss1  = F.binary_cross_entropy_with_logits(out5, label_224) + iou_loss(out5, label_224)
        #lossp  = F.binary_cross_entropy_with_logits(pose, label_224) + iou_loss(pose, label_224)

        loss = loss_weights[0] * loss1 + loss_weights[1] * loss2 + loss_weights[2] * loss3 + loss_weights[3] * loss4

        #img_total_loss = loss_weights[0] * loss1 + loss_weights[1] * loss3 + loss_weights[2] * loss4 + loss_weights[3] * loss5\
                       # +loss_weights[0] * loss1c + loss_weights[1] * loss3c + loss_weights[2] * loss4c + loss_weights[3] * loss5c
        #contour_total_loss = loss_weights[0] * c_loss1 + loss_weights[1] * c_loss3 + loss_weights[2] * c_loss4 + loss_weights[3] * c_loss5\
                           # +loss_weights[0] * c_loss1c + loss_weights[1] * c_loss3c + loss_weights[2] * c_loss4c + loss_weights[3] * c_loss5c

        #total_loss = loss# + contour_total_loss# + loss1c

        opt.zero_grad()
        loss.backward()
        #with amp.scale_loss(loss, opt) as scale_loss:
        #        scale_loss.backward()
        opt.step()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        epoch_total_loss += loss.cpu().data.item()
        epoch_loss1 += loss1.cpu().data.item()
        epoch_loss2 += loss2.cpu().data.item()
        epoch_loss3 += loss3.cpu().data.item()
        epoch_loss4 += loss4.cpu().data.item()
        #epoch_lossp += lossp.cpu().data.item()
        #epoch_loss41 += loss41.cpu().data.item()
        progress_bar.set_postfix(loss=f'{epoch_loss1/(i+1):.3f}')
    return epoch_loss1/l#, epoch_loss41/l
        
def fit(epochs, model, lr, train_dl, method):
    #progress_bar = tqdm(range(epochs))#,desc='Epoch[{:03d}/{:03d}]'.format(epochs, 40))
    f = 'loss.txt'
    step = len(epochs)
    for st in range(step):
        opt = get_opt(lr,model)

        for epoch in range(epochs[st]):

            model.train()
            loss = train_one_epoch(epoch,epochs[st],model,opt,train_dl)
            #model.eval()
            #val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
            #print(val_loss)
            with open(f, 'a') as fe:
                fe.write(str(sum(epochs[:st])+epoch+1)+'\t{loss:.3f}\n'.format(loss = loss))
                               # fe.write(str(sum(epochs[:st])+epoch+1)+'\t{loss1:.3f}\t{loss2:.3f}\t{loss3:.3f}\t{loss4:.3f}\t{lossp:.3f}\n'.format(loss1 = loss[0],loss2 = loss[1],loss3 = loss[2],loss4 = loss[3],lossp = loss[4]))

            #writer.add_scalar('Loss/train', loss, sum(epochs[:st])+epoch+1)

            #progress_bar.set_postfix(loss=f'{val_loss:.3f}')
            if epoch % 20 == 19:
                torch.save(model.state_dict(),"savepth/tmp/"+str(sum(epochs[:st])+epoch+1))
                get_pred_dir(model)
                thread = threading.Thread(target = eval)
                thread.start()
                #evaluate(method=method)

        lr = lr/10
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