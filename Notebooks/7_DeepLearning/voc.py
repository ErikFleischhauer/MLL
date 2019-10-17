import os

import numpy as np
import scipy.io as sio
import torch
from skimage.io import imread
from PIL import Image
from torch.utils import data
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import KMeans

num_classes = 21
ignore_label = 255

#root = 'V:/VOC/'
root = '/work2/HaMLeT_datasets/VOC/'

'''
color map
0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle # 6=bus, 7=car, 8=cat, 9=chair, 10=cow, 11=diningtable,
12=dog, 13=horse, 14=motorbike, 15=person # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
'''

class_to_label = {0: 'background', 1:'aeroplane', 2:'bicycle', 3:'bird', 4:'boat', 5:'bottle', 6:'bus', 7:'car', 8:'cat', 9:'chair', 10:'cow', 11:'diningtable',
12:'dog', 13:'horse', 14:'motorbike', 15:'person', 16:'potted plant', 17:'sheep', 18:'sofa', 19:'train', 20:'tv/monitor'}

palette = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
           (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
           (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)]

for i in range(256 - len(palette)):
    palette.append((0, 0, 0))

decolorizer = KMeans(n_clusters=21)
decolorizer.cluster_centers_ = np.asarray(palette[:21])


def colorize_mask(mask):
    new_mask = [palette[px] for px in mask.flatten()]
    new_mask = np.uint8(np.reshape(new_mask, mask.shape+(3,)))
    return new_mask


def make_dataset(mode):
    assert mode in ['train', 'valid', 'test']
    items = []
    if mode == 'train':
        img_path = os.path.join(root, 'benchmark_RELEASE', 'dataset', 'img')
        mask_path = os.path.join(root, 'benchmark_RELEASE', 'dataset', 'cls')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'benchmark_RELEASE', 'dataset', 'train.txt')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.mat'))
            items.append(item)
    elif mode == 'valid':
        img_path = os.path.join(root, 'VOCdevkit', 'VOC2012', 'JPEGImages')
        mask_path = os.path.join(root, 'VOCdevkit', 'VOC2012', 'SegmentationClass')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'VOCdevkit', 'VOC2012', 'ImageSets', 'Segmentation', 'seg11valid.txt')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.png'))
            items.append(item)
    else:
        img_path = os.path.join(root, 'VOCdevkit (test)', 'VOC2012', 'JPEGImages')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'VOCdevkit (test)', 'VOC2012', 'ImageSets', 'Segmentation', 'test.txt')).readlines()]
        for it in data_list:
            items.append((img_path, it))
    return items


class VOC(data.Dataset):
    def __init__(self, mode, transform=None, target_transform=None):
        super(VOC, self).__init__()
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([list(range(1, 21)),])
        self.imgs = make_dataset(mode)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index, colorize=False):
        if self.mode == 'test':
            img_path, img_name = self.imgs[index]
            img = np.uint8(imread(os.path.join(img_path, img_name + '.jpg'))[:,:,:3])
            if self.transform is not None:
                img = self.transform(img)
            return img_name, img

        img_path, mask_path = self.imgs[index]
        img = np.uint8(imread(img_path)[:, :, :3])
        if self.mode == 'train':
            #@Dbug: Hier wird die GT geladen, müsste dann ['CategoriesPresent'] oder so sein
            matdata = sio.loadmat(mask_path)
            mask = matdata['GTcls']['Segmentation'][0][0]
            classes = matdata['GTcls']['CategoriesPresent'][0][0].flatten()
            mask = colorize_mask(mask) if colorize else self.target_transform(Image.fromarray(np.uint8(mask)))
                
        elif self.mode == 'valid':
            mask_img = np.uint8(imread(mask_path)[:, :, :3])
            mask = np.reshape(decolorizer.predict(mask_img.reshape(-1, 3)), mask_img.shape[:2])
            uniques = np.unique(mask)
            mask = mask_img if colorize else self.target_transform(Image.fromarray(np.uint8(mask)))
            classes = uniques[(uniques != 255) & (uniques != 0)]

        else:
            raise Exception('Unrecognized Mode {}'.format(self.mode))

        if self.transform is not None:
            img = self.transform(Image.fromarray(img))

        return img, mask, np.float32(self.mlb.transform([classes,])[0])

    def __len__(self):
        return len(self.imgs)