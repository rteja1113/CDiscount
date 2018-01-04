#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 20:01:42 2017

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

def pytorch_image_to_tensor_transform(image):
    mean = [0.485, 0.456, 0.406 ]
    std  = [0.229, 0.224, 0.225 ]
    #image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    #image = image.transpose((2,0,1))
    #tensor = torch.from_numpy(image).float().div(255)
    tensor[0] = (tensor[0] - mean[0]) / std[0]
    tensor[1] = (tensor[1] - mean[1]) / std[1]
    tensor[2] = (tensor[2] - mean[2]) / std[2]
    return tensor

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


def train_model(model, dataloaders, criterion, optimizer, dataset_sizes, num_epochs=1):
    since = time.time()

    #best_model_wts = model.state_dict()
    best_acc = 0.0

#for epoch in range(num_epochs):
#    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
#    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            #scheduler.step()
            model.train(True)  # Set model to training mode
        else:
            model.train(False)  # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for i, data in enumerate(dataloaders[phase]):
            # get the inputs
            inputs, labels = data
            #labels = labels.long()
            #print(type(labels))    
            # wrap them in Variable
            if True:
                if phase == 'train':
                    inputs = Variable(inputs).cuda()
                    labels = Variable(labels).cuda()
                else :
                    inputs = Variable(inputs, volatile = True).cuda()
                    labels = Variable(labels).cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
            #print(type(labels))
            # forward
            #outputs = torch.nn.DataParallel(model)(inputs)
            #logits = torch.nn.DataParallel(model)(inputs)
            logits = model(inputs)
            probs = F.softmax(logits)
            
            #outputs = F.softmax(logits)
            #_, preds = torch.max(outputs.data, 1)
            #loss = criterion(outputs, labels)
            loss = F.cross_entropy(logits, labels)
 
            # backward + optimize only if in training phase
            if phase == 'train':
                loss.backward()
                optimizer.step()
            
            acc   = top_accuracy(probs, labels, top_k=(1,))
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


        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects / dataset_sizes[phase]

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))

        # deep copy the model
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            #best_model_wts = model.state_dict()

#    print()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
    torch.save(model.state_dict(), '/home/cvpr/Documents/kaggle/data/cdisc/pytorch_models/resnext_101_64/val_acc_{:.5f}.pth'.format(best_acc))
    
    # load best model weights
    #model.load_state_dict(best_model_wts)
    return
    

if __name__ == "__main__":
    net = resnext_101_64x4d.resnext_101_64x4d
    net.load_state_dict(torch.load("/home/cvpr/Documents/convert_torch_to_pytorch/resnext_101_64x4d.pth"))
    net = torch.nn.Sequential(*list(net.children())[:-1])
    net.add_module('final', torch.nn.Linear(2048, 5270))
    net.load_state_dict(torch.load('/home/cvpr/Documents/kaggle/data/cdisc/pytorch_models/resnext_101_64/val_acc_0.66436.pth'))    

#    for i, c in enumerate(net.children()):
#        if i>=7:
#            for p in c.parameters():
#                p.requires_grad=True
#        else:
#            for p in c.parameters():
#                p.requires_grad=False
        
    
    for i, c in enumerate(net.children()):
        if i>=7:
            for p in c.parameters():
                p.requires_grad=True
        else:
            for p in c.parameters():
                p.requires_grad=False
    
    for i, c in enumerate(net.children()):
        if i==6:
            for j, gc in enumerate(c.children()):
                if j>=10:
                    for p in gc.parameters():
                        p.requires_grad=True
    
    
    
    
    
    train_df = pd.read_pickle("/media/cvpr/ssd/cdisc/train_offsets_0.9.pkl")
    val_df = pd.read_pickle("/media/cvpr/ssd/cdisc/valid_offsets_0.1.pkl")
    idx2cat = pickle.load(open("/media/cvpr/ssd/cdisc/idx2cat.pkl", "rb"))
    
    data_transforms = {
                       'train': transforms.Compose([
                                transforms.Scale(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                #lambda x: pytorch_image_to_tensor_transform(x),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]),
                       'val': transforms.Compose([
                              transforms.Scale(224),
                              transforms.ToTensor(),
                              #lambda x: pytorch_image_to_tensor_transform(x)
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ]),}
                       

    
    datasets = {'train':CdiscDataset('/media/cvpr/ssd/cdisc/train', train_df, 
                                     idx2cat, transform=data_transforms['train'], with_labels=True), 
                'val':CdiscDataset('/media/cvpr/ssd/cdisc/valid', val_df, 
                                     idx2cat, transform=data_transforms['val'], with_labels=True)}
    
    dataloaders = {'train':DataLoader(datasets['train'], batch_size=64, shuffle=True, num_workers=4, pin_memory=True), 
                   'val':DataLoader(datasets['val'], batch_size=64, shuffle=False, num_workers=4, pin_memory=True)}

    dataset_sizes = {x:len(datasets[x]) for x in ["train", "val"]}
    #optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.001, momentum=0.9, nesterov=True)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0001)
    #net = torch.nn.DataParallel(net).cuda()
    net.cuda()
    #criterion = torch.nn.CrossEntropyLoss().cuda()
    train_model(net, dataloaders, None, optimizer, dataset_sizes)
    torch.save(net.state_dict(), '/home/cvpr/Documents/kaggle/data/cdisc/pytorch_models/resnext_101_64/train_acc_0.5916.pth')    
    
    
    