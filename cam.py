from multiscale_fusion_sod import SOD
import torch
from matplotlib import pyplot as plt
from dataloader import get_loader
import numpy as np
import torch.nn.functional as F
import cv2
from PIL import Image
import transforms as trans
from torchvision import transforms
def iou_loss(pred, mask):
    pred  = torch.sigmoid(pred)
    inter = (pred*mask).sum(dim=(2,3))
    union = (pred+mask).sum(dim=(2,3))
    iou  = 1-(inter+1)/(union-inter+1)
    return iou.mean()

model = SOD()
#model.cuda()
model.eval()
model.load_state_dict(torch.load('/home/yy/savepth/multiscale_fusion_sod_d2_int4d_cpr120.pth'))
train_dataset = get_loader('DUTS/DUTS-TE', "/home/yy/datasets/", 224, mode='test')
train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle = False, 
                                               pin_memory=True,num_workers = 1)
path = ''
gt = None
img_size = 14
for i,batch in enumerate(train_dl):
    l = []

    images, image_w, image_h, image_path = batch
    path = image_path
    a1,a11,a12,a2,a21,a22,a3,a31 = model(images)
    l.append(a3.clone().detach().cpu().numpy())
    l.append(a31.clone().detach().cpu().numpy())
    #l.append(a12.clone().detach().cpu().numpy())

    if(i == 6):
        break

print(path)


x1 = np.sum(l[0],axis=2)
x1 = x1.reshape(1,img_size,img_size).squeeze(0)
#print(x1)

mask = x1
mask = mask - np.min(mask)
mask = mask / np.max(mask)
colormap = cv2.COLORMAP_JET
use_rgb = True
img = cv2.imread(path[0])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.array(img).astype(np.float32)
img = img/255
heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
if use_rgb:
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
heatmap = np.float32(heatmap) / 255
heatmap = cv2.resize(heatmap,(img.shape[1],img.shape[0]),interpolation=cv2.INTER_LINEAR)
if np.max(img) > 1:
    raise Exception(
        "The input image should np.float32 in the range [0, 1]")

cam = 0.3*heatmap + 0.7*img
cam = cam / np.max(cam)
plt.imshow(cam)
plt.savefig('cam.png')
cv2.imwrite('cam.png',cv2.cvtColor(cam*255, cv2.COLOR_BGR2RGB))

x1 = np.sum(l[1],axis=2)
x1 = x1.reshape(1,img_size,img_size).squeeze(0)
#print(x1)

mask = x1
mask = mask - np.min(mask)
mask = mask / np.max(mask)
colormap = cv2.COLORMAP_JET
use_rgb = True
img = cv2.imread(path[0])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.array(img).astype(np.float32)
img = img/255
heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
if use_rgb:
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
heatmap = np.float32(heatmap) / 255
heatmap = cv2.resize(heatmap,(img.shape[1],img.shape[0]),interpolation=cv2.INTER_LINEAR)
if np.max(img) > 1:
    raise Exception(
        "The input image should np.float32 in the range [0, 1]")

cam = 0.3*heatmap + 0.7*img
cam = cam / np.max(cam)
plt.imshow(cam)
plt.savefig('cam1.png')
cv2.imwrite('cam1.png',cv2.cvtColor(cam*255, cv2.COLOR_BGR2RGB))

ll=l[1]-l[0]
print(ll.shape)
x1 = np.sum(l[2],axis=2)
x1 = x1.reshape(1,img_size,img_size).squeeze(0)
#print(x1)

mask = x1
mask = mask - np.min(mask)
mask = mask / np.max(mask)
colormap = cv2.COLORMAP_JET
use_rgb = True
img = cv2.imread(path[0])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.array(img).astype(np.float32)
img = img/255
heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
if use_rgb:
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
heatmap = np.float32(heatmap) / 255
heatmap = cv2.resize(heatmap,(img.shape[1],img.shape[0]),interpolation=cv2.INTER_LINEAR)
if np.max(img) > 1:
    raise Exception(
        "The input image should np.float32 in the range [0, 1]")

cam = 0.3*heatmap + 0.7*img
cam = cam / np.max(cam)
plt.imshow(cam)
plt.savefig('cam2.png')
cv2.imwrite('cam2.png',cv2.cvtColor(cam*255, cv2.COLOR_BGR2RGB))

print(1)