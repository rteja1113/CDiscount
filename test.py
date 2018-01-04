#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 14:24:26 2017

@author: cvpr
"""

from cdisc_kutils import *
from cdisc_models import *
import glob
import functools
from keras import layers
from keras.models import Sequential, Model, load_model
from keras.preprocessing.image import load_img, img_to_array
pd.options.display.max_rows = 999
seed = 2017
np.set_printoptions(threshold=np.inf, precision=4)
np.random.seed(seed)

if __name__ == '__main__':
    # loading data files
    meta_dir = '../data/cdisc/'
    test_offsets_df = pd.read_pickle(os.path.join(data_dir, "test_offsets.pkl"))
    test_images_df = pd.read_pickle(os.path.join(data_dir, "test_imgs_df.pkl"))
    test_bson_file = open(os.path.join(data_dir, "test.bson"), "rb")
    class_order = pickle.load(open(meta_dir+'class_order_kalic.pkl', 'rb'))
    idx2cat = pickle.load(open(os.path.join(data_dir, "idx2cat.pkl"), "rb"))
    
    num_classes = len(idx2cat)
    img_size = 180
    batch_size = 256
    
    
    #models
    inception = load_model(meta_dir+'models/inception_v2_kalic.hdf5')
    xception = load_model(meta_dir+'models/xception_v2_kalic.hdf5')
        
    
    model_base = Models(img_size, num_classes)
    resnet_model = model_base.resnet101()
    resnet_model.load_weights(meta_dir+'models/resnet101/res4b14_branch2a_val_acc_0.6660_epoch13.h5')
    
    
    
    for l in inception.layers:
        l.trainable = False
    
    for l in xception.layers:
        l.trainable = False
        
    for l in resnet_model.layers:
        l.trainable = False
    
    
    inception.compile('sgd', loss='categorical_crossentropy')
    xception.compile('sgd', loss='categorical_crossentropy')    
    resnet_model.compile('sgd', loss='categorical_crossentropy')        
    # test datagen
    test_datagen = ImageDataGenerator()
    test_lock= threading.Lock()
    test_gen = BSONIterator(test_bson_file, test_images_df, test_offsets_df, 
                         len(class_order), test_datagen, test_lock, target_size=(img_size, img_size),
                         batch_size=256, shuffle=False, with_labels=False)
    
    scores = predict_generator([inception, xception, resnet_model], test_gen, steps=math.ceil(test_images_df.shape[0]/batch_size), workers=4, out_file=None, verbose=1)
    #scores = predict_generator([inception, xception, ], test_gen, steps=521, workers=4, out_file=None, verbose=1)
    scores_df = pd.DataFrame.from_dict(scores, orient='index')
    scores_df = scores_df.reset_index()
    scores_df.rename(columns={'index':'_id', 0:'category_id'}, inplace=True)
    scores_df['category_id'] = scores_df['category_id'].apply(lambda x: idx2cat[x])
    scores_df.to_csv('../data/cdisc/submissions/inc_xcp_resnet101_flipped.csv', index=False)
    
    
    