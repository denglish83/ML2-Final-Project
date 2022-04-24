#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 12:24:59 2020

@author: thomaskeeley
"""

import os

wd = '/home/ubuntu/project/'

# path where pre-trained weights should be saved
model_dir = 'keras-retinanet/snapshots/'
# model url
model_url = 'https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet50_coco_best_v2.1.0.h5'

# retrieve model from model url and save
os.chdir(wd + model_dir)
os.system('wget ' + model_url)