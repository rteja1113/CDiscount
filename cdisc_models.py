#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 00:44:06 2017

@author: cvpr
"""
from cdisc_resnet50 import *
from cdisc_resnet101 import *
from cdisc_inception_resnet_v2 import *
import json
class Models:
    def __init__(self, input_shape, classes):
        self.input_shape = input_shape
        self.classes = classes

    
    def resnet50(self):
        resnet_base = ResNet50(include_top=False, input_shape=(self.input_shape, self.input_shape, 3), pooling='avg')
        x = Dense(self.classes, activation='softmax', name='fc{}'.format(self.classes), input_shape=resnet_base.output_shape)(resnet_base.output)
        resnet_model = Model(resnet_base.input, x)
        return resnet_model
    
    def resnet101(self):
        resnet_base = resnet101_model('/home/cvpr/.keras/models/resnet101_weights_tf.h5', input_shape=(self.input_shape, self.input_shape, 3))
        last_layer = resnet_base.get_layer('avg_pool')
        x = Dense(self.classes, activation='softmax', name='fc{}'.format(self.classes), input_shape=last_layer.output_shape)(last_layer.output)
        resnet_model = Model(resnet_base.input, x)
        return resnet_model
        
        #with open('/home/cvpr/.keras/models/imagenet_class_index.json') as f:
        #    classes = json.load(f)
            
        
        #img = cv2.imread('../data/cdisc/rottie3.jpg')
        #img = cv2.resize(img, (224, 224))
        #pred = resnet_base.predict(np.expand_dims(img, 0))
        #classes[str(pred.argmax())]
        
    def inception_resnet(self):
        inc_res_base = InceptionResNetV2(include_top=False, input_shape=(self.input_shape, self.input_shape, 3), pooling='avg')
        #inc_res_base = InceptionResNetV2(include_top=False, input_shape=(180, 180, 3), pooling='avg')
        x = Dense(self.classes, activation='softmax', name='fc{}'.format(self.classes), input_shape=inc_res_base.output_shape)(inc_res_base.output)
        #x = Dense(5270, activation='softmax', name='fc{}'.format(5270), input_shape=inc_res_base.output_shape)(inc_res_base.output)
        inc_res_model = Model(inc_res_base.input, x)
        return inc_res_model