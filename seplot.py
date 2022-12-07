from multiscale_fusion_sod import SOD
import torch
from matplotlib import pyplot as plt
from dataloader import get_loader
import numpy as np
model = SOD()
model.cuda()
model.eval()
model.load_state_dict(torch.load('/home/yy/savepth/multiscale_fusion_sod_d2_int6_se_cpr120.pth'))
train_dataset = get_loader('DUTS/DUTS-TE', "/home/yy/datasets/", 224, mode='test')
train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle = True, 
                                               pin_memory=True,num_workers = 4)

l = []
ll1 = []
ll2 = []

for i,batch in enumerate(train_dl):
    images, image_w, image_h, image_path = batch

    w1,w2 = model(images.cuda())
    w1 = w1.squeeze(2).squeeze(2).squeeze(0)
    w2 = w2.squeeze(2).squeeze(2).squeeze(0)
    ll1.append(w1.clone().detach().cpu().numpy())
    ll2.append(w2.clone().detach().cpu().numpy())
    if(i == 200):
        break
cnt = 200
l1 = []
l2 = []
l3 = []
l4 = []
for i in range(0,cnt):
    l1.append(np.sum(ll1[i][:]))
    l2.append(np.sum(ll2[i][:]))

ll = np.concatenate([l1,l2],axis=0)
x = np.arange(1,401)
size = np.ones(401)
plt.scatter(x,ll,sizes=size)
plt.savefig('se.png')