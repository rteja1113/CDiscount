#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 17:23:02 2017

@author: cvpr
"""

import os
#os.environ['CUDA_VISIBLE_DEVICES'] =
#NUM_CUDA_DEVICES = len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) 
import glob
import time
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import pickle
from torch import optim
from temp_pytorch_utils import *
import torch.nn as nn
import hashlib
import sys
sys.path.append('../../convert_torch_to_pytorch') 
import resnext_101_64x4d
import h5py
import collections
from tqdm import tqdm
CDISCOUNT_NUM_CLASSES = 5270



def top_accuracy(probs, labels, top_k=(1,)):
    """Computes the precision@k for the specified values of k"""

    probs  = probs.data
    labels = labels.data

    max_k = max(top_k)
    batch_size = labels.size(0)

    values, indices = probs.topk(max_k, dim=1, largest=True,  sorted=True)
    indices  = indices.t()
    corrects = indices.eq(labels.view(1, -1).expand_as(indices))

    accuracy = []
    for k in top_k:
        # https://stackoverflow.com/questions/509211/explain-slice-notation
        # a[:end]      # items from the beginning through end-1
        c = corrects[:k].view(-1).float().sum(0, keepdim=True)
        #accuracy.append(c.mul_(1. / batch_size))
        accuracy.append(c)
    return accuracy

def train(model, loader, optimizer, dataset_size, phase):
    since = time.time()
    best_acc = 0.0
    if phase == "train":
        model.train(True)
    else:
        model.train(False)
    running_loss = 0.0
    running_corrects = 0.0
    
    for i, (x, y, product_id) in enumerate(loader):
        x = Variable(x).cuda()
        y = Variable(y).cuda()
        optimizer.zero_grad()
        logits = model(x)
        probs = F.softmax(logits)
        loss = F.cross_entropy(logits, y)
        
        if phase == "train":
            loss.backward()
            optimizer.step()
        
        acc   = top_accuracy(probs, y, top_k=(1,))
        batch_acc  = acc[0][0]
        batch_loss = loss.data[0]    
        # statistics
        running_loss += batch_loss
        running_corrects += batch_acc
        
        if (i%1000 == 999) & (phase== 'train'):
            print('batch-{}: {} Loss: {:.4f} Acc: {:.4f}'.format(
                  i,phase, running_loss/((i+1)*64), running_corrects/((i+1)*64)))
        
        #if (i > 30000):
        #    break


    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects / dataset_size

    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        phase, epoch_loss, epoch_acc))





if __name__ == "__main__":
    data_dir = "/media/cvpr/ssd/cdisc/"
    meta_dir = "/home/cvpr/Documents/kaggle/data/cdisc/submissions/chunk_preds/inception/"
    
    idx2cat = pickle.load(open(data_dir+"idx2cat.pkl", "rb"))

    val_df = pd.read_pickle("/media/cvpr/ssd/cdisc/valid_offsets_0.1.pkl")
    val_df.drop_duplicates("product_id", inplace=True)
    val_df["product_id"] = val_df["product_id"].astype(np.int32)
    val_df["category_idx"] = val_df["category_idx"].astype(np.int32)
    val_df["category_idx"] = val_df["category_idx"].astype(np.int32)
    
    logit_dataset = LogitDataset(val_df, data_dir+"val_logits/", idx2cat, transforms.Compose([lambda x: torch.FloatTensor(x)]), True)    
    logit_loader = DataLoader(logit_dataset, batch_size=64, shuffle=True, num_workers=4)
    
    model = nn.Sequential(nn.BatchNorm1d(CDISCOUNT_NUM_CLASSES),
                          nn.Linear(CDISCOUNT_NUM_CLASSES, 512),
                          nn.BatchNorm1d(512), 
                          nn.ReLU(),
                          nn.Linear(512, CDISCOUNT_NUM_CLASSES))
    
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    model.cuda()
    for i in range(5):
        train(model, logit_loader, optimizer, len(val_df), "train")
    
    
    
from pytho
    