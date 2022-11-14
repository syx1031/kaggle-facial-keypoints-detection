# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

from fastai.vision.all import AffineCoordTfm, Pipeline, TensorImage, TensorPoint, affine_mat
import matplotlib.pyplot as plt
import torch
import random

from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
import torch.nn as nn
from torch.nn import functional as F
from torch import optim

print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.__version__)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

train_csv = pd.read_csv('../input/facial-keypoints-detection/training.zip')
train_csv_images = train_csv['Image']
train_csv = train_csv.drop(columns='Image')
train_csv.info()

test_csv = pd.read_csv('../input/facial-keypoints-detection/test.zip')
test_csv_images = test_csv['Image']
test_csv = test_csv.drop(columns='Image')
test_csv.info()

train_images = [np.fromstring(train_csv_images.iloc[i], sep=' ').reshape([96,96]) for i in range(train_csv_images.size)]
train_points = [train_csv.iloc[k].values.reshape([15,2]) for k in range(train_csv.shape[0]) ]
test_images = [np.fromstring(test_csv_images[i], sep=' ').reshape([96,96]) for i in range(test_csv_images.size)]

# plt.imshow(train_images[0], cmap='gray')
# plt.plot(train_points[0][:,0], train_points[0][:,1], 'gx')

# plt.imshow(test_images[0], cmap='gray')

def augment(img, pnts, rot_deg, zoom_factor, x_shift_pix, y_shift_pix):
    sz = img.shape[-2:]
    def get_rotation(x):
        mysz = x.new_ones(x.shape[0])
        rot_rad = torch.ones_like(mysz)*(rot_deg / 180.0 * np.pi)
        m11 = rot_rad.cos() / zoom_factor
        m12 = rot_rad.sin() / zoom_factor
        t0 = torch.ones_like(mysz)*(x_shift_pix/48.0)
        t1 = torch.ones_like(mysz)*(y_shift_pix/48.0)
        return affine_mat(m11, m12, t0, -m12, m11, t1)
    t1 = AffineCoordTfm(aff_fs=get_rotation, size=sz)
    p1 = Pipeline(funcs=t1)
    x = TensorImage(img).view([1,1,96,96])
    y = TensorPoint(pnts, img_size=[96,96]).view([1,15,2])
    x,y = p1((x,y/48.0-1.0))
    y = y.view([15,2])
    coord_ok = (y[:,0] > -1.0) & (y[:,0] < 1.0) & (y[:,1] > -1.0) & (y[:,1] < 1.0)
    coord_ok = torch.stack([coord_ok, coord_ok], dim=1)
    y = y.where(coord_ok, torch.tensor(np.nan))
    y = y*48.0+48.0
    return np.array(x.view([96,96])), np.array(y)

augs = []
one_pixel = 2.0/96.0
for dx in range(21):
    for dy in range(21):
        if dx==10 and dy==10:
            continue
        augs.append([0.0, 1.0, dx-10.0, dy-10.0])
for rot_deg in range(21):
    if rot_deg==10:
        continue
    augs.append([rot_deg-10, 1.0, 0.0, 0.0])
for scale in range(21):
    if scale==10:
        continue
    augs.append([0.0, 0.9 + 0.01*scale, 0.0, 0.0])

aug_images = []
aug_points = []
aug_ind = 0
for k1 in range(len(train_images)):
    img,pnt = augment(train_images[k1], train_points[k1], *augs[aug_ind])
    aug_images.append(train_images[k1])
    aug_points.append(train_points[k1])
    aug_images.append(img)
    aug_points.append(pnt)
    aug_ind = (aug_ind + 1) % len(augs)
    
train_images = aug_images
train_points = aug_points

class My_MSELoss(nn.Module):
    def __init__(self):
        super(My_MSELoss, self).__init__()
    def forward(self, inp, targ):
        if targ.dtype!=torch.float16:
            targ = targ.float()
        tmptarg = targ.view(inp.shape)
        tmptarg = torch.where(torch.isnan(tmptarg), inp, tmptarg)
        func = nn.MSELoss(reduction='mean')
        return func(inp, tmptarg)

class face_data(torch.utils.data.Dataset):
    def __init__(self, images, points=None, train=True):
        if points:
            for i in range(len(images)):
                images[i] = images[i].astype(np.float32)
                images[i] = np.expand_dims(images[i], 0)
                points[i] = points[i].astype(np.float32)
        else:
            for i in range(len(images)):
                images[i] = images[i].astype(np.float32)
                images[i] = np.expand_dims(images[i], 0)
        
        self.images = images
        self.points = points
        self.train = train
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        if self.train:
            return self.images[index].reshape((1, 96, 96)), self.points[index]
        else:
            return self.images[index].reshape((1, 96, 96))
        
train_dataset = face_data(train_images, points=train_points, train=True)
test_dataset = face_data(test_images, points=None, train=False)

class Train_Sampler(Sampler):
    def __init__(self, dataset):
        len = dataset.__len__()
        point = int(len * 0.8)
        self.indices = list(range(point))
    def __iter__(self):
        random.shuffle(self.indices)
        return iter(self.indices)
    def __len__(self):
        return len(self.indices)

class Val_Sampler(Sampler):
    def __init__(self, dataset):
        len = dataset.__len__()
        point = int(len * 0.8)
        self.indices = [k for k in range(point, len) if k % 2 == 0]
    def __iter__(self):
        random.shuffle(self.indices)
        return iter(self.indices)
    def __len__(self):
        return len(self.indices)

train_sampler = Train_Sampler(train_dataset)
val_sampler = Val_Sampler(train_dataset)

batch_size = 64

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False, sampler=train_sampler, num_workers=0)
val_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False, sampler=val_sampler, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, drop_last=False, num_workers=0)


class RestNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.re1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)


    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x + output)


class RestNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(stride[0], stride[0]), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.re1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(stride[1], stride[1]), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(stride[0], stride[0]), bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.bn1(output))

        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(extra_x + out)

class AdaptiveConcatPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool_one = nn.AdaptiveAvgPool2d(output_size=1)
        self.pool_two = nn.AdaptiveMaxPool2d(output_size=1)

    def forward(self, x):
        return torch.cat([self.pool_one(x), self.pool_two(x)], 1)


class RestNet18(nn.Module):
    def __init__(self):
        super(RestNet18, self).__init__()
        self.layer0 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
                                    nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False))

        self.layer1 = nn.Sequential(RestNetBasicBlock(64, 64, (1, 1)),
                                    RestNetBasicBlock(64, 64, (1, 1)))

        self.layer2 = nn.Sequential(RestNetDownBlock(64, 128, [2, 1]),
                                    RestNetBasicBlock(128, 128, (1, 1)))

        self.layer3 = nn.Sequential(RestNetDownBlock(128, 256, [2, 1]),
                                    RestNetBasicBlock(256, 256, (1, 1)))

        self.layer4 = nn.Sequential(RestNetDownBlock(256, 512, [2, 1]),
                                    RestNetBasicBlock(512, 512, (1, 1)))
        self.layer5 = nn.Sequential(AdaptiveConcatPool2d(), 
                                    nn.Flatten(),
                                    nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.Dropout(p=0.25, inplace=False),
                                    nn.Linear(in_features=1024, out_features=512, bias=False),
                                    nn.ReLU(inplace=True),
                                    nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.Dropout(p=0.5, inplace=False),
                                    nn.Linear(in_features=512, out_features=30, bias=False))

    def forward(self, x):
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out

model = RestNet18().to(device)

# for n in model.modules():
#     print(n)

criteon = My_MSELoss().to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-2)

for epoch in range(100):
    model.train()
    L = 0
    batch = 0
    for image, point in train_dataloader:
        image, point = image.to(device), point.to(device)

        logits = model(image)
        loss = criteon(logits, point)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        L = L + loss.item()
        batch = batch + 1

    print(epoch, 'train_loss:', L / batch)

    model.eval()
    with torch.no_grad():
        L = 0
        batch = 0
        for image, point in val_dataloader:
            image, point = image.to(device), point.to(device)

            logits = model(image)
            loss = criteon(logits, point)

            L = L + loss.item()
            batch = batch + 1
        
        print(epoch, 'val_loss:', L / batch)

print('finetune...')

model.train()

for layer in model.modules():
    if isinstance(layer, nn.BatchNorm2d):
        layer.eval()
        for param in layer.parameters():
            param.requires_grad = False
    if isinstance(layer, nn.BatchNorm1d):
        layer.eval()
        for param in layer.parameters():
            param.requires_grad = False
    if isinstance(layer, nn.Dropout):
        layer.eval()
        for param in layer.parameters():
            param.requires_grad = False

finetune_optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=1e-3, weight_decay=1e-5, momentum=0.9)

for epoch in range(50):
    L = 0
    batch = 0
    for image, point in val_dataloader:
        image, point = image.to(device), point.to(device)

        logits = model(image)
        loss = criteon(logits, point)

        finetune_optimizer.zero_grad()
        loss.backward()
        finetune_optimizer.step()
        
        L = L + loss.item()
        batch = batch + 1

    print(epoch, 'val_loss:', L / batch)


test_predictions = np.zeros([0, 30])
for images in test_dataloader:
    model.eval()
    images = images.to(device)
    preds = model(images)
    preds = torch.where(preds>=96, torch.ones(preds.shape).to(device)*95.99, preds)
    preds = torch.where(preds<=0, torch.ones(preds.shape).to(device)*0.01, preds)
    test_predictions = np.concatenate((test_predictions, preds.detach().cpu().numpy()), axis=0)


lut = pd.read_csv('../input/facial-keypoints-detection/IdLookupTable.csv', index_col='RowId')
sample = pd.read_csv('../input/facial-keypoints-detection/SampleSubmission.csv', index_col='RowId')
sample['Location'] = sample['Location'].astype(np.float32)
namedict = {train_csv.columns[k1]: k1 for k1 in range(len(train_csv.columns))}
for k1 in range(sample.shape[0]):
    imageid = lut.iloc[k1]['ImageId'] - 1
    featurename = lut.iloc[k1]['FeatureName']
    featurecol = namedict[featurename]
    sample.iloc[k1]['Location'] = test_predictions[imageid, featurecol]

sample.to_csv('/kaggle/working/submission.csv')


print('done')