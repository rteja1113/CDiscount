#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 20:50:39 2017

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
import hashlib
import sys
sys.path.append('../../convert_torch_to_pytorch') 
import resnext_101_64x4d
import h5py
import collections
from tqdm import tqdm
CDISCOUNT_NUM_CLASSES = 5270

def test_model(net, dataloader, test_df, dname, f,num_crops=5):
    
    #sub_dict =  collections.OrderedDict()
    #for p_id in test_df.product_id.unique():
    #    sub_dict[str(p_id)] = np.zeros(CDISCOUNT_NUM_CLASSES, dtype=np.float32) 
        
    for c in range(num_crops):
        for batch_x1, _, batch_product_ids in tqdm(dataloader):
            batch_product_ids = batch_product_ids.numpy()
            if c > 5:
                idx = [i for i in range(batch_x1.size()[3]-1, -1, -1)]
                idx = torch.LongTensor(idx)
                batch_x1 = batch_x1.index_select(3, idx)
            batch_x1 = Variable(batch_x1, volatile=True).cuda()
            #probs = F.softmax(net(batch_x1))
            logits = net(batch_x1)
            logits  = logits.cpu().data.numpy()
            for i,pred in enumerate(logits):
                #if pred.max() > sub_dict[str(batch_product_ids[i])].max():
                np.save(dname+str(batch_product_ids[i])+".npy", pred)
                    
                
    return

if __name__ == "__main__":
    net = resnext_101_64x4d.resnext_101_64x4d
    net = torch.nn.Sequential(*list(net.children())[:-1])
    net.add_module('final', torch.nn.Linear(2048, 5270))
    net.load_state_dict(torch.load('/home/cvpr/Documents/kaggle/data/cdisc/pytorch_models/resnext_101_64/val_acc_0.68468.pth'))    
    data_dir = "/media/cvpr/ssd/cdisc/"
    net.cuda().eval()
    idx2cat = pickle.load(open(data_dir+"idx2cat.pkl", "rb"))
    meta_dir = "/home/cvpr/Documents/kaggle/data/cdisc/logits/"
    #file_handle = h5py.File(data_dir+"resnext_101_64_valid_logits.h5", "w")
    
    #train_df = pd.read_pickle("/media/cvpr/ssd/cdisc/train_offsets_0.9.pkl")
    val_df = pd.read_pickle("/media/cvpr/ssd/cdisc/valid_offsets_0.1.pkl")
    
    data_transforms = {
#                       'train': transforms.Compose([
#                                transforms.Scale(224),
#                                transforms.RandomHorizontalFlip(),
#                                transforms.ToTensor(),
#                                #lambda x: pytorch_image_to_tensor_transform(x),
#                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#                    ]),
                       'val': transforms.Compose([
                              transforms.Scale(224),
                              transforms.ToTensor(),
                              #lambda x: pytorch_image_to_tensor_transform(x)
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ]),}
                       

    
    datasets = {#'train':CdiscDataset('/media/cvpr/ssd/cdisc/train', train_df, 
                #                     idx2cat, transform=data_transforms['train'], with_labels=True), 
                'val':CdiscDataset('/media/cvpr/ssd/cdisc/valid', val_df, 
                                     idx2cat, transform=data_transforms['val'], with_labels=True)}
    
    dataloaders = {#'train':DataLoader(datasets['train'], batch_size=64, shuffle=True, num_workers=4, pin_memory=True), 
                   'val':DataLoader(datasets['val'], batch_size=256, shuffle=False, num_workers=4, pin_memory=True)}

    
    test_model(net, dataloaders["val"], val_df, data_dir + "val_logits/", None, 1)    
    file_handle.close()
    
    
    
    
    file_handle =  h5py.File(data_dir+"resnext_101_64_valid_logits.h5", "r")
    
    for dset in file_handle:
        np.save(meta_dir+"numpys/" + dset + ".npy", dset[:])
        
    
    
    
    
    
    
    