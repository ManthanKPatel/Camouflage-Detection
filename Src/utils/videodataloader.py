import os
from PIL import Image, ImageEnhance
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
#import time
from glob import glob
import os.path as osp
#import pdb
#from mypath import Path


# several data augumentation strategies
def cv_random_flip(imgs, label):
    # left right flip
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        for i in range(len(imgs)):
            imgs[i] = imgs[i].transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return imgs, label


def randomCrop(imgs, label):
    border = 30
    image_width = imgs[0].size[0]
    image_height = imgs[0].size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)

    for i in range(len(imgs)):
        imgs[i] = imgs[i].crop(random_region)
    return imgs, label.crop(random_region)


def randomRotation(imgs, label):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        for i in range(len(imgs)):
            imgs[i] = imgs[i].rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
    return imgs, label


def colorEnhance(imgs):
    for i in range(len(imgs)):
        bright_intensity = random.randint(5, 15) / 10.0
        imgs[i] = ImageEnhance.Brightness(imgs[i]).enhance(bright_intensity)
        contrast_intensity = random.randint(5, 15) / 10.0
        imgs[i] = ImageEnhance.Contrast(imgs[i]).enhance(contrast_intensity)
        color_intensity = random.randint(0, 20) / 10.0
        imgs[i] = ImageEnhance.Color(imgs[i]).enhance(color_intensity)
        sharp_intensity = random.randint(0, 30) / 10.0
        imgs[i] = ImageEnhance.Sharpness(imgs[i]).enhance(sharp_intensity)
    return imgs


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])

    for i in range(noiseNum):
        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:
            img[randX, randY] = 0
        else:
            img[randX, randY] = 255
    return Image.fromarray(img)


class VideoDataset(data.Dataset):
    def __init__(self, data_root, trainsize):
        self.trainsize = trainsize
        #self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        #self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
        #            or f.endswith('.png')]
        #self.images = sorted(self.images)
        #self.gts = sorted(self.gts)
        self.images = []
        self.gts = []

        for scene in os.listdir(osp.join(data_root)):
            image = sorted(glob(osp.join(data_root, scene, 'Imgs', '*.jpg')))
            gt_list = sorted(glob(osp.join(data_root, scene, 'GT', '*.png')))
            self.images += image
            self.gts += gt_list
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        image = self.img_transform(image)
        #image, gt = randomRotation(image, gt)
        gt = self.gt_transform(gt)
        #image = colorEnhance(image)
        #gt = randomPeper(gt)
        return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


# dataloader for training
def get_vloader(data_root, batchsize, trainsize,
               shuffle=True, num_workers=12, pin_memory=True):
    dataset = VideoDataset(data_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

class test_dataset:
    """load test dataset (batchsize=1)"""
    def __init__(self, data_root, testsize):
        self.testsize = testsize

        self.images = []
        self.gts = []
        self.filename = []
        self.scene_list = []

        for scene in os.listdir(osp.join(data_root)):
            print("Sacene name:", scene)
            # file = sorted([name for name in os.listdir(scene)])
            image = sorted(glob(osp.join(data_root, scene, 'Imgs', '*.jpg')))
            gt_list = sorted(glob(osp.join(data_root, scene, 'GT', '*.png')))
            file = sorted(os.listdir(osp.join(data_root,scene,'Imgs')))
            self.images += image
            self.gts += gt_list
            self.filename += file
            for i in range(len(image)):
                self.scene_list.append(str(scene))
        # self.filter_files()
        self.size = len(self.images)

        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.filename[self.index]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        # name = self.filename[self.index]
        scene = self.scene_list[self.index]
        self.index += 1
        return image, gt, name, scene

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def apply_mask(self, image, mask, color, alpha=0.5):
        """Apply the given mask to the image.
        """
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c] * 255,
                                      image[:, :, c])
        return image

    def __len__(self):
        return self.size
