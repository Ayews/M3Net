#import pathlib
import numpy as np
from PIL import Image
from torch.utils import data
import transforms as trans
from torchvision import transforms
import random
import os
def get_path(x = 10000):
    data_root = "/mnt/disk2/dataset/MSRA10K_Imgs_GT"
    img_root = data_root + '/train/'
    img_files = os.listdir(img_root)
    images = []
    labels = []
    for img in img_files:

        images.append(img_root + img[:-4]+'.jpg')
        labels.append(img_root.replace('/train/', '/truth/') + img[:-4]+'.png')
        #contours.append(img_root.replace('/DUTS-TR-Image/', '/DUTS-TR-Contour/') + img[:-4] + '.png')

    return images[0:x], labels[0:x]#, contours

'''
dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
    

def preprocess(x, y):
    return x.view(-1, 3, 224, 224).to(dev), y.view(-1,1,224,224).to(dev)



def get_data(img_paths,truth_paths):
    x_train = []
    for path in img_paths:
        img = cv2.imread(path)
        img = cv2.resize(img,(224,224))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = img*1.0/255
        x_train.append(img)
    y_train = []
    for path in truth_paths:
        img = cv2.imread(path,0)
        img = cv2.resize(img,(224,224))
        img = img*1.0/255
        y_train.append(img)
    x_train = torch.from_numpy(np.asarray(x_train).astype(np.float32).transpose(0,3,1,2))
    y_train = torch.from_numpy(np.asarray(y_train).astype(np.float32))
    #print(x_train.shape,y_train.shape)
    train_ds = TensorDataset(x_train, y_train)
    #valid_ds = TensorDataset(x_train[9000:10000], y_train[9000,10000])
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    train_dl = WrappedDataLoader(train_dl, preprocess)
    return train_dl
'''
def load_list(dataset_name, data_root):

    images = []
    labels = []
    #contours = []

    img_root = data_root + dataset_name + '/imgs/'
    img_files = os.listdir(img_root)

    for img in img_files:

        images.append(img_root + img[:-4]+'.jpg')
        labels.append(img_root.replace('/imgs/', '/gt/') + img[:-4]+'.png')
        #contours.append(img_root.replace('/imgs/', '/DUTS-TR-Contour/') + img[:-4] + '.png')

    return images, labels#, contours


def load_test_list(test_path, data_root):

    images = []

    if 'DUTS' in test_path:
        img_root = data_root + test_path + '\\DUTS-TE-Image\\'
    else:
        img_root = data_root + test_path + '\\imgs\\'

    img_files = os.listdir(img_root)
    if '\\HKU-IS\\' in img_root:
        ext = '.png'
    else:
        ext = '.jpg'
    for img in img_files:
        images.append(img_root + img[:-4] + ext)

    return images


class ImageData(data.Dataset):
    def __init__(self, dataset_list, data_root, transform, mode, img_size=None, scale_size=None, t_transform=None, label_14_transform=None, label_28_transform=None, label_56_transform=None, label_112_transform=None):

        if mode == 'train':
            #self.image_path, self.label_path = load_list(dataset_list, data_root)
            self.image_path, self.label_path = load_list(dataset_list, data_root)
        else:
            self.image_path = load_test_list(dataset_list, data_root)

        self.transform = transform
        self.t_transform = t_transform
        self.label_14_transform = label_14_transform
        self.label_28_transform = label_28_transform
        self.label_56_transform = label_56_transform
        self.label_112_transform = label_112_transform
        self.mode = mode
        self.img_size = img_size
        self.scale_size = scale_size

    def __getitem__(self, item):
        fn = self.image_path[item].split('/')

        filename = fn[-1]

        image = Image.open(self.image_path[item]).convert('RGB')
        image_w, image_h = int(image.size[0]), int(image.size[1])

        if self.mode == 'train':

            label = Image.open(self.label_path[item]).convert('L')
            random_size = self.scale_size

            new_img = trans.Scale((random_size, random_size))(image)
            new_label = trans.Scale((random_size, random_size), interpolation=Image.NEAREST)(label)

            # random crop
            w, h = new_img.size
            if w != self.img_size and h != self.img_size:
                x1 = random.randint(0, w - self.img_size)
                y1 = random.randint(0, h - self.img_size)
                new_img = new_img.crop((x1, y1, x1 + self.img_size, y1 + self.img_size))
                new_label = new_label.crop((x1, y1, x1 + self.img_size, y1 + self.img_size))

            # random flip
            if random.random() < 0.5:
                new_img = new_img.transpose(Image.FLIP_LEFT_RIGHT)
                new_label = new_label.transpose(Image.FLIP_LEFT_RIGHT)

            new_img = self.transform(new_img)

            label_14 = self.label_14_transform(new_label)
            label_28 = self.label_28_transform(new_label)
            label_56 = self.label_56_transform(new_label)
            label_112 = self.label_112_transform(new_label)
            label_224 = self.t_transform(new_label)


            return new_img, label_224, label_14, label_28, label_56, label_112,
        else:

            image = self.transform(image)

            return image, image_w, image_h, self.image_path[item]

    def __len__(self):
        return len(self.image_path)


def get_loader(dataset_list, data_root, img_size, mode='train'):

    if mode == 'train':

        transform = trans.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        t_transform = trans.Compose([
            transforms.ToTensor(),
        ])
        label_14_transform = trans.Compose([
            trans.Scale((img_size // 16, img_size // 16), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        label_28_transform = trans.Compose([
            trans.Scale((img_size//8, img_size//8), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        label_56_transform = trans.Compose([
            trans.Scale((img_size//4, img_size//4), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        label_112_transform = trans.Compose([
            trans.Scale((img_size//2, img_size//2), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        if img_size == 224:
            scale_size = 256
        else:
            scale_size = 512
    else:
        transform = trans.Compose([
            trans.Scale((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 处理的是Tensor
        ])

    if mode == 'train':
        dataset = ImageData(dataset_list, data_root, transform, mode, img_size, scale_size, t_transform, label_14_transform, label_28_transform, label_56_transform, label_112_transform)
    else:
        dataset = ImageData(dataset_list, data_root, transform, mode)

    # data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_thread)
    return dataset