import os
import torch
import torch.nn as nn
from data.dataloader import RGB_Dataset
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
from M3Net import M3Net

def get_pred_dir(model, data_root = '/home/yy/datasets/', save_path = 'preds/',img_size = 384,methods = 'DUT-O+DUTS+ECSSD+HKU-IS+PASCAL-S+SOD'):
    batch_size = 1
    test_paths = methods.split('+')
    for dataset_setname in test_paths:
        test_dataset = RGB_Dataset(data_root, [dataset_setname], img_size,'test')
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        progress_bar = tqdm(test_loader, desc=dataset_setname,ncols=140)
        for i,data_batch in enumerate(progress_bar):
            images = data_batch['image']
            image_w,image_h = data_batch['shape']
            image_w, image_h = int(image_w[0]), int(image_h[0])
            image_path = data_batch['name']
            images = Variable(images.cuda())

            outputs_saliency = model(images)
            mask_1_1 = outputs_saliency[-1]
            pred = torch.sigmoid(mask_1_1)

            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((image_h, image_w))
            ])

            pred = pred.squeeze(0)
            pred = transform(pred)

            filename = image_path[0]
            # save saliency maps
            save_test_path = save_path+dataset_setname+'/'
            if not os.path.exists(save_test_path):
                os.makedirs(save_test_path)
            pred.save(os.path.join(save_test_path, filename + '.png'))

def testing(args):
    print('Starting test.')
    model = M3Net(embed_dim=384,dim=64,img_size=args.img_size,method=args.method)
    model.cuda()
    model.load_state_dict(torch.load(args.save_model+args.method+'.pth'))
    print('Loaded from '+args.save_model+args.method+'.pth.')
    model.eval()
    get_pred_dir(model,data_root=args.data_root,save_path=args.save_test,img_size = args.img_size,methods=args.test_methods)
    print('Predictions are saved at '+args.save_test+'.')
