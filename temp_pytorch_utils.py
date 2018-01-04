from __future__ import print_function, division
import os, sys, math, io, warnings, gc
import pandas as pd
import bson
import struct
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
#import cv2
from PIL import Image

def img_to_array(img, data_format=None):
    x = np.asarray(img, dtype=np.float32)
    return x


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')



class CdiscDataset(Dataset):


    def __init__(self, directory, images_df, idx2cat, transform=None, with_labels=True):
        self.images_df = images_df
        self.idx2cat = idx2cat
        self.num_class = len(idx2cat)
        self.with_labels = with_labels
        self.transform = transform
        self.directory = directory

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self, idx):
        image_row = self.images_df.iloc[idx]
        product_id = image_row["product_id"]
        img_idx = image_row["img_idx"]
        if self.with_labels:
            category_idx = image_row["category_idx"]
            f = self.directory + '/' + str(self.idx2cat[category_idx]) + '/' + str(product_id) \
                + '-' + str(img_idx) + '.jpg'
        else:
            f = self.directory + '/' + str(product_id) + '-' + str(img_idx) + '.jpg'
        x = pil_loader(f)
        
        #for t in self.transform:
        #    x = t(x)
        if self.transform:
            x = self.transform(x)
        
        if self.with_labels:
            #y = np.zeros(self.num_class)
            #y[image_row["category_idx"]] = 1
            y = image_row["category_idx"]
            return x, y, product_id
        else:
            return x, product_id


class LogitDataset(Dataset):


    def __init__(self, images_df, folder, idx2cat, transform=None, with_labels=True):
        self.images_df = images_df
        self.idx2cat = idx2cat
        self.num_class = len(idx2cat)
        self.with_labels = with_labels
        self.transform = transform
        self.folder = folder

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self, idx):
        image_row = self.images_df.iloc[idx]
        product_id = image_row["product_id"]
        
        
        
        
        x = np.load(self.folder + '/' + str(product_id) + ".npy")
        
        if self.transform:
            x = self.transform(x)
        
        if self.with_labels:
            #y = np.zeros(self.num_class)
            #y[image_row["category_idx"]] = 1
            y = image_row["category_idx"]
            return x, y, product_id
        else:
            return x, product_id

#train_dataset = CdiscDataset(data_dir+"/train", train_images_df,
#                             idx2cat,
#                             transform=transforms.Compose([transforms.RandomSizedCrop(160),
#                                     transforms.RandomHorizontalFlip(),
#                                     transforms.ToTensor(),
#                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
#
#train_loader = DataLoader(train_dataset, 256, shuffle=True, num_workers=8)
#counter = 0
#
#
#for data in train_loader:
#    inputs, labels = data
#    print("inputs size: {}, labels size: {}".format(inputs.size(), labels.size()))
#    print(type(inputs), type(labels))
#    counter+=1
#    if counter == 10:
#        break




