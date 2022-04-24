#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 08:37:48 2020

@author: thomaskeeley
"""
import os
import sys
import glob
import pandas as pd
import random
from PIL import Image
import cv2
# %% -------------------------------------------------------------------------------------------------------------------
os.getcwd()
wd = '/home/ubuntu/data/'
os.chdir(wd)

os.chdir(wd + 'images/')
img_file_list = [i for i in glob.iglob('*.png')]

os.chdir(wd + 'annotations/')
annot_file_list = [i for i in glob.iglob('*.txt')]

# %% -------------------------------------------------------------------------------------------------------------------
print("Reading and formatting annotations...")
vehicle_images = []
annot_df = pd.DataFrame(columns=['path', 'x_min', 'y_min', 'x_max', 'y_max', 'object'])
for idx, file in enumerate(annot_file_list):
    sys.stdout.write('\r{}%  '.format(round((idx / len(annot_file_list))*100, 3)))
    img_file = file.replace('txt', 'png')
    if img_file in img_file_list:
        with open(file, "r") as text:
            lines = text.readlines()
            for line in lines[2:]:
                split = line.split(' ')

                if split[-2] == 'small-vehicle':
                    filename = '/home/ubuntu/data/images/{}'.format(img_file)
                    vehicle_images.append(filename)
                    x_min = min(split[:-2][0::2])
                    x_max = max(split[:-2][0::2])
                    y_min = min(split[:-2][1::2])
                    y_max = max(split[:-2][1::2])

                    annot_df = annot_df.append({'path': filename, 'x_min': int(float(x_min)), 'y_min': int(float(y_min)),
                                                'x_max': int(float(x_max)), 'y_max': int(float(y_max)),
                                                'object': split[-2]}, ignore_index=True)

# %% -------------------------------------------------------------------------------------------------------------------
print("Slicing images and refactoring annotations...")
img_slices = pd.DataFrame(columns=['path', 'x_min', 'y_min', 'x_max', 'y_max', 'object'])
for idx, path in enumerate(annot_df['path'].unique()):
    sys.stdout.write('\r{}%  '.format(round((idx / len(annot_df['path'].unique())*100), 3)))
    annots = annot_df.loc[annot_df['path'] == path]
    img = cv2.imread(path)
    img2 = img
    height, width, channels = img.shape
    crop_h = round((height / 500) / 2) * 2
    crop_w = round((width / 500) / 2) * 2

    for ih in range(crop_h):
        for iw in range(crop_w):
            x = int(width / crop_w * iw)
            y = int(height / crop_h * ih)
            h = int(height / crop_h)
            w = int(width / crop_w)

            img = img[y:y + h, x:x + w]

            filename = '/home/ubuntu/data/image_slices/{}_{}_{}'.format(ih, iw, path.split('/')[-1])
            cv2.imwrite(filename, img)
            img = img2
            for i in range(len(annots)):
                line = annots.iloc[i]
                if (
                        x <= int(line['x_min']) and x + w >= int(line['x_max']) and
                        y <= int(line['y_min']) and y + h >= int(line['y_max'])
                ):

                    x_min = int(line['x_min'] - x)
                    x_max = int(line['x_max'] - x)
                    y_min = int(line['y_min'] - y)
                    y_max = int(line['y_max'] - y)

                    img_slices = img_slices.append({'path': filename, 'x_min': x_min, 'y_min': y_min,
                                                    'x_max': x_max, 'y_max': y_max, 'object': 'small-vehicle'},
                                                     ignore_index=True)


# %% -------------------------------------------------------------------------------------------------------------------
print("Cleaning annotations...")
os.chdir(wd + 'image_slices/')
for file in img_slices['path'].unique():
    im = Image.open(file)
    x_max, y_max = im.size
    for idx in img_slices.loc[img_slices['path'] == file].index:
        if img_slices.iloc[idx]['x_max'] > x_max:
            img_slices.iloc[idx]['x_max'] = x_max
        if img_slices.iloc[idx]['y_max'] > y_max:
            img_slices.iloc[idx]['y_max'] = y_max

img_slices = (img_slices[(img_slices['x_max'] > img_slices['x_min']) & (img_slices['y_max'] > img_slices['y_min'])]).reset_index(drop=True)

# %% -------------------------------------------------------------------------------------------------------------------
vehicle_images = set(img_slices['path'])
n_test = int(len(vehicle_images) * 0.1)

random.seed(42)
test_images = random.sample(vehicle_images, n_test)
train_images = [i for i in vehicle_images if i not in test_images]
val_images = random.sample(train_images, n_test)
train_images = [i for i in train_images if i not in val_images]

train_df = img_slices[img_slices['path'].isin(train_images)]
test_df = img_slices[img_slices['path'].isin(test_images)]
val_df = img_slices[img_slices['path'].isin(val_images)]
class_df = pd.DataFrame(columns=['class', 'id']).append({'class': 'small-vehicle', 'id':0}, ignore_index=True)
# %% -------------------------------------------------------------------------------------------------------------------
class_df.to_csv(wd + 'classes.csv', header=False, index=False)
train_df.to_csv(wd + 'train.csv', header=False, index=False)
val_df.to_csv(wd + 'val.csv', header=False, index=False)
test_df.to_csv(wd + 'test.csv', header=False, index=False)
# %% -------------------------------------------------------------------------------------------------------------------
