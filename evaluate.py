
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 17:53:58 2020
@author: thomaskeeley


Early stopping (lines

"""
import sys
import os

os.system('sudo pip install torch')
os.system('sudo pip install torchvision')
os.system('sudo pip install pycocotools')
os.system('sudo pip install scikit-image')
os.system('sudo pip install Pillow')
os.system('sudo pip install tifffile')
os.system('sudo pip install rasterio')
import rasterio
import numpy as np
import pandas as pd
import glob
import torch
import torch.nn
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

wd = '/home/ubuntu/project/'

os.chdir(wd + '/pytorch-retinanet/retinanet/')
sys.path.append('/home/ubuntu/final_project/Project/pytorch-retinanet/')

from retinanet.dataloader import *
from retinanet import csv_eval

# %%
# Import the trained model and compile the validation dataset

# model = torch.load(wd + 'pytorch-retinanet/csv_retinanet_27.pt')
model = torch.load(wd + 'model_final5.pt')

os.chdir(wd)
dataset_val = CSVDataset(train_file='/home/ubuntu/data/test10.csv',
                         class_list='/home/ubuntu/data/classes.csv',
                         transform=transforms.Compose([Normalizer(), Resizer()]))

# %%
# Compile the detections of the test images and calculate the mAP

dets = csv_eval._get_detections(dataset_val, model, score_threshold=0.5, max_detections=1000, save_path=None)

# average_precision = csv_eval.evaluate(dataset_val, model)
test_df = pd.read_csv('/home/ubuntu/data/test10.csv', header=None)

img_path_list = (test_df[0].unique()).tolist()

# %%
# Ingest the test images and show the predicted bounding boxes

for i in range(len(img_path_list)):
    path = img_path_list[i]
    detections = dets[i][0]
    img = cv2.imread(path)

    plt.figure()
    plt.imshow(img)
    for bbox in detections:
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])

        box = Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red')
        plt.axes().add_patch(box)

    plt.show()

# %%
# Compare the various models and show the inference time as well as mAP

# model_names = ['retinanet_res18.pt', 'retinanet_res50.pt', 'retinanet_res101.pt']
model_names = ['model_final5.pt', 'retinanet_res18.pt', 'model_final_12.pt']

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

for mod in model_names:
    model = torch.load(wd + mod)
    start.record()
    csv_eval.evaluate(dataset_val, model)
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()

    print(mod, start.elapsed_time(end) / len(dataset_val))

# %%
# Read in the satellite images for further test, compile a csv file formatted for testing

os.chdir('/home/ubuntu/project/sat_image/')
sat_images = [i for i in glob.iglob('*.tif')]

sat_images_df = pd.DataFrame(columns=[1, 2, 3, 4, 5, 6])
for file in sat_images:
    sat_images_df = sat_images_df.append({1: file}, ignore_index=True)

sat_images_df.to_csv('/home/ubuntu/data/sat_images_df.csv', header=False, index=False)

# %%
# Compile the satellite image testing data and conduct the detections using the trained model

dataset_val = CSVDataset(train_file='/home/ubuntu/data/sat_images_df.csv',
                         class_list='/home/ubuntu/data/classes.csv',
                         transform=transforms.Compose([Normalizer(), Resizer()]))

dets = csv_eval._get_detections(dataset_val, model, score_threshold=0.00, max_detections=1000, save_path=None)

# %%
# Ingest the satellite images and show the predicted bounding boxes

for i in range(len(sat_images)):
    path = sat_images[i]
    detections = dets[i][0]
    if len(detections > 0):
        img = cv2.imread(path)

        plt.figure()
        plt.imshow(img)
        for bbox in detections:
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])

            box = Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red')
            plt.axes().add_patch(box)

        plt.show()

# %%
# Reformat the annotations to match the original geographic context of the satellite image, convert to well known text

wkt_strings = []
for i in range(len(sat_images)):
    path = sat_images[i]
    detections = dets[i][0]
    image = rasterio.open(path)
    crs = str(image.crs).split(':')[1]
    im_width = image.width
    im_height = image.height
    left, bottom, right, top = image.bounds
    im_x_res = (right - left) / im_width
    im_y_res = (top - bottom) / im_height

    for bbox in detections:
        x_min = (int(bbox[0]) * im_x_res) + left
        y_min = top - (int(bbox[1]) * im_y_res)
        x_max = (int(bbox[2]) * im_x_res) + left
        y_max = top - (int(bbox[3]) * im_y_res)

        wkt = 'POLYGON (({} {}, {} {}, {} {}, {} {}, {} {}))'.format(
            x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max, x_min, y_min)
        wkt_strings.append(wkt)

df = pd.DataFrame(wkt_strings, columns=['wkt'])
df.to_csv(wd + 'obj_det_wkt-{}.csv'.format(crs))

