# Pseudocode
# For each image in validation set
#   run prediction
#   read in labels
#   for each label
#       find predicted boxes that have same class
#       Calculate overlap
#       keep max overlap
# Calculate average overlap per label
# calculate average overlap (mAP)

# Code edited from https://github.com/postor/DOTA-yolov3/blob/master/test.py

import cv2
import numpy as np
import time
import glob

# read class names from text file
classes = None
with open('/home/ubuntu/home/ubuntu/project/DOTA-yolov3/cfg/dota.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# function to get the output layer names in the architecture

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1]
                     for i in net.getUnconnectedOutLayers()]
    return output_layers


def detect(image, net):
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    # create input blob
    blob = cv2.dnn.blobFromImage(
        image, scale, (416, 416), (0, 0, 0), True, crop=False)

    # set input blob for the network
    net.setInput(blob)

    # run inference through the network
    # and gather predictions from output layers
    outs = net.forward(get_output_layers(net))

    # initialization
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    # for each detetion from each output layer
    # get the confidence, class id, bounding box params
    # and ignore weak detections (confidence < 0.5)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # apply non-max suppression
    indices = cv2.dnn.NMSBoxes(
        boxes, confidences, conf_threshold, nms_threshold)

    return boxes, indices

# taken from retinanet/csv_eval
def compute_overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua

#My code
net = cv2.dnn.readNet('/home/ubuntu/home/ubuntu/project/darknet/backup/dota-yolov3-tiny_900.weights', '/home/ubuntu/home/ubuntu/project/DOTA-yolov3/cfg/dota-yolov3-tiny.cfg')
img_list = glob.glob('/home/ubuntu/home/ubuntu/project/DOTA-yolov3/dataset/val/images/*.png')
start_time = time.time()
counts = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
match_counts = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
overlap = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
for img in img_list:
    print(img)
    image = cv2.imread(img)
    boxes, labels = detect(image, net)
    truth = None
    str = '/home/ubuntu/home/ubuntu/project/DOTA-yolov3/dataset/val/labelTxt/'+img[64:69]+'.txt'
    with open(str, 'r') as f:
        truth = [line.strip().split(' ') for line in f.readlines()]
        truth.pop(0)
        truth.pop(0)
    for x in range(len(truth)):
        if truth[x][8] == 'small-vehicle':
            counts[0] += 1
        elif truth[x][8] == 'large-vehicle':
            counts[1] += 1
        elif truth[x][8] == 'plane':
            counts[2] += 1
        elif truth[x][8] == 'storage-tank':
            counts[3] += 1
        elif truth[x][8] == 'ship':
            counts[4] += 1
        elif truth[x][8] == 'harbor':
            counts[5] += 1
        elif truth[x][8] == 'ground-track-field':
            counts[6] += 1
        elif truth[x][8] == 'soccer-ball-field':
            counts[7] += 1
        elif truth[x][8] == 'tennis-court':
            counts[8] += 1
        elif truth[x][8] == 'swimming-pool':
            counts[9] += 1
        elif truth[x][8] == 'baseball-diamond':
            counts[10] += 1
        elif truth[x][8] == 'roundabout':
            counts[11] += 1
        elif truth[x][8] == 'basketball-court':
            counts[12] += 1
        elif truth[x][8] == 'bridge':
            counts[13] += 1
        elif truth[x][8] == 'helicopter':
            counts[14] += 1
        elif truth[x][8] == 'container-crane':
            counts[15] += 1

        true_box = [[min(float(truth[x][0]), float(truth[x][2]), float(truth[x][4]), float(truth[x][6]))],
                    [min(float(truth[x][1]), float(truth[x][3]), float(truth[x][5]), float(truth[x][7]))],
                    [max(float(truth[x][0]), float(truth[x][2]), float(truth[x][4]), float(truth[x][6]))],
                    [max(float(truth[x][1]), float(truth[x][3]), float(truth[x][5]), float(truth[x][7]))]]
        true_box_array = np.transpose(np.array(true_box))

        max_ov = 0
        for y in range(len(labels)):
            a = boxes[y][0]
            b = boxes[y][1]
            w = boxes[y][2]
            h = boxes[y][3]

            x1 = a
            x2 = b
            x3 = a+w
            x4 = b+h
            box = [[x1], [x2], [x3], [x4]]
            box_array = np.transpose(np.array(box))

            if truth[x][8]=='small-vehicle':
               if labels[y]==0:
                    calc = compute_overlap(true_box_array, box_array)
               else:
                   calc = 0
            elif truth[x][8] == 'large-vehicle':
                if labels[y] == 1:
                    calc = compute_overlap(true_box_array, box_array)
                else:
                    calc = 0
            elif truth[x][8] == 'plane':
                if labels[y] == 2:
                    calc = compute_overlap(true_box_array, box_array)
                else:
                    calc = 0
            elif truth[x][8] == 'storage-tank':
                if labels[y] == 3:
                    calc = compute_overlap(true_box_array, box_array)
                else:
                    calc = 0
            elif truth[x][8] == 'ship':
                if labels[y] == 4:
                    calc = compute_overlap(true_box_array, box_array)
                else:
                    calc = 0
            elif truth[x][8] == 'harbor':
                if labels[y] == 5:
                    calc = compute_overlap(true_box_array, box_array)
                else:
                    calc = 0
            elif truth[x][8] == 'ground-track-field':
                if labels[y] == 6:
                    calc = compute_overlap(true_box_array, box_array)
                else:
                    calc = 0
            elif truth[x][8] == 'soccer-ball-field':
                if labels[y] == 7:
                    calc = compute_overlap(true_box_array, box_array)
                else:
                    calc = 0
            elif truth[x][8] == 'tennis-court':
                if labels[y] == 8:
                    calc = compute_overlap(true_box_array, box_array)
                else:
                    calc = 0
            elif truth[x][8] == 'swimming-pool':
                if labels[y] == 9:
                    calc = compute_overlap(true_box_array, box_array)
                else:
                    calc = 0
            elif truth[x][8] == 'baseball-diamond':
                if labels[y] == 10:
                    calc = compute_overlap(true_box_array, box_array)
                else:
                    calc = 0
            elif truth[x][8] == 'roundabout':
                if labels[y] == 11:
                    calc = compute_overlap(true_box_array, box_array)
                else:
                    calc = 0
            elif truth[x][8] == 'basketball-court':
                if labels[y] == 12:
                    calc = compute_overlap(true_box_array, box_array)
                else:
                    calc = 0
            elif truth[x][8] == 'bridge':
                if labels[y] == 13:
                    calc = compute_overlap(true_box_array, box_array)
                else:
                    calc = 0
            elif truth[x][8] == 'helicopter':
                if labels[y] == 14:
                    calc = compute_overlap(true_box_array, box_array)
                else:
                    calc = 0
            elif truth[x][8] == 'container-crane':
                if labels[y] == 15:
                    calc = compute_overlap(true_box_array, box_array)
                else:
                    calc = 0


            if calc > max_ov:
                max_ov = calc

        if max_ov > 0:
            if truth[x][8] == 'small-vehicle':
                match_counts[0] += 1
                overlap[0] += max_ov
            elif truth[x][8] == 'large-vehicle':
                match_counts[1] += 1
                overlap[1] += max_ov
            elif truth[x][8] == 'plane':
                match_counts[2] += 1
                overlap[2] += max_ov
            elif truth[x][8] == 'storage-tank':
                match_counts[3] += 1
                overlap[3] += max_ov
            elif truth[x][8] == 'ship':
                match_counts[4] += 1
                overlap[4] += max_ov
            elif truth[x][8] == 'harbor':
                match_counts[5] += 1
                overlap[5] += max_ov
            elif truth[x][8] == 'ground-track-field':
                match_counts[6] += 1
                overlap[6] += max_ov
            elif truth[x][8] == 'soccer-ball-field':
                match_counts[7] += 1
                overlap[7] += max_ov
            elif truth[x][8] == 'tennis-court':
                match_counts[8] += 1
                overlap[8] += max_ov
            elif truth[x][8] == 'swimming-pool':
                match_counts[9] += 1
                overlap[9] += max_ov
            elif truth[x][8] == 'baseball-diamond':
                match_counts[10] += 1
                overlap[10] += max_ov
            elif truth[x][8] == 'roundabout':
                match_counts[11] += 1
                overlap[11] += max_ov
            elif truth[x][8] == 'basketball-court':
                match_counts[12] += 1
                overlap[12] += max_ov
            elif truth[x][8] == 'bridge':
                match_counts[13] += 1
                overlap[13] += max_ov
            elif truth[x][8] == 'helicopter':
                match_counts[14] += 1
                overlap[14] += max_ov
            elif truth[x][8] == 'container-crane':
                match_counts[15] += 1
                overlap[15] += max_ov

AP = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
match_rate = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
sum_AP = 0
for i in range(len(AP)):
    if counts[i] >0:
        AP[i] = overlap[i]/counts[i]
    else:
        AP[i] = 0
    sum_AP += AP[i]

    if counts[i] >0:
        match_rate[i] = match_counts[i]/counts[i]
    else:
        match_rate[i] = 0

mAP = sum_AP/16

use_time = time.time() - start_time
print(counts)
print(match_rate)
print(AP)
print(mAP)
print(use_time)