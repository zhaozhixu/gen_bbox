#! /usr/bin/python

import numpy as np
import re
import os
import sys
import time
from ctypes import *

INPUT_C = 3
INPUT_H = 368
INPUT_W = 640

CONVOUT_C = 144
CONVOUT_H = 23
CONVOUT_W = 40

CLASS_SLICE_C = 99
CONF_SLICE_C = 9
BBOX_SLICE_C = 36

OUTPUT_CLS_SIZE = 11
OUTPUT_BBOX_SIZE = 4

TOP_N_DETECTION = 64
NMS_THRESH = 0.4
# PROB_THRESH = 0.005
PROB_THRESH = 0.3
PLOT_PROB_THRESH = 0.4
# EPSILON = 1e-16

ANCHORS_PER_GRID = 9
# ANCHOR_SHAPE = [36, 37, 366, 174, 115, 59,
#                 162, 87, 38, 90, 258, 173,
#                 224, 108, 78, 170, 72, 43]

H, W, B = CONVOUT_H, CONVOUT_W, ANCHORS_PER_GRID
anchor_shapes = np.reshape(
    [np.array(
        [[229, 137], [48, 71], [289, 245],
         [185, 134], [85, 142], [31, 41],
         [197, 191], [237, 206], [63, 108]])] * H * W,
    (H, W, B, 2)
)
center_x = np.reshape(
    np.transpose(
        np.reshape(
            np.array([np.arange(1, W+1)*float(INPUT_W)/(W+1)]*H*B),
            (B, H, W)
        ),
        (1, 2, 0)
    ),
    (H, W, B, 1)
)
center_y = np.reshape(
    np.transpose(
        np.reshape(
            np.array([np.arange(1, H+1)*float(INPUT_H)/(H+1)]*W*B),
            (B, W, H)
        ),
        (2, 1, 0)
    ),
    (H, W, B, 1)
)
anchors = np.reshape(
    np.concatenate((center_x, center_y, anchor_shapes), axis=3),
    (-1, 4)
)
ANCHORS = anchors

# CLASS_NAMES = ["car", "pedestrian", "cyclist"]
# CLASS_NAMES = ["car", "person", "riding", "bike_riding", "boat", "truck", "horse_riding"]
CLASS_NAMES = ["person", "car", "riding", "boat", "drone", "truck", "parachute", "whale", "building", "bird", "horse_riding"]

E = 2.718281828

def parse_tensor_str(tensor_str):
    strre = r'(?<=\d)(?=\s)'
    parsed_str = re.sub(strre, ',', tensor_str)
    strre = r'\](\s*)\['
    m = re.search(strre, parsed_str)
    if m:
        parsed_str = re.sub(strre, '],' + m.group(1) + '[', parsed_str)
    tensor = eval(parsed_str)
    return tensor

def safe_exp(w):
    if w < 1:
        return np.exp(w)
    return w * E

def transform_bbox(bbox_delta, anchor, img_width, img_height):
    x_scale = 1.0 * img_width / INPUT_W
    y_scale = 1.0 * img_height / INPUT_H
    delta_x = bbox_delta[0][0]
    delta_y = bbox_delta[0][1]
    delta_w = bbox_delta[0][2]
    delta_h = bbox_delta[0][3]
    anchor_x = anchor[0][0]
    anchor_y = anchor[0][1]
    anchor_w = anchor[0][2]
    anchor_h = anchor[0][3]
    cx = (anchor_x + delta_x * anchor_w) * x_scale
    cy = (anchor_y + delta_y * anchor_h) * y_scale
    w = anchor_w * safe_exp(delta_w) * x_scale
    h = anchor_w * safe_exp(delta_h) * y_scale
    xmin = np.min([np.max([cx-w*0.5, 0]), img_width-1])
    ymin = np.min([np.max([cy-h*0.5, 0]), img_height-1])
    xmax = np.max([np.min([cx+w*0.5, img_width-1]), 0])
    ymax = np.max([np.min([cy+h*0.5, img_height-1]), 0])

    return np.array([xmin, ymin, xmax, ymax])

def feature_express(feature_map, img_width, img_height):
    # convout = feature_map[:, 1:-1, 1:-1]
    convout = feature_map
    # class_feature = convout[0:CLASS_SLICE_C]
    conf_feature = convout[CLASS_SLICE_C:CLASS_SLICE_C+CONF_SLICE_C]
    bbox_feature = convout[CLASS_SLICE_C+CONF_SLICE_C:CLASS_SLICE_C+CONF_SLICE_C+BBOX_SLICE_C]
    conf_reshape1 = np.reshape(conf_feature, (ANCHORS_PER_GRID, 1, CONVOUT_H, CONVOUT_W))
    bbox_reshape1 = np.reshape(bbox_feature, (ANCHORS_PER_GRID, OUTPUT_BBOX_SIZE, CONVOUT_H, CONVOUT_W))
    conf_reshape2 = np.reshape(np.transpose(conf_reshape1, (2, 3, 0, 1)), (-1, 1))
    bbox_reshape2 = np.reshape(np.transpose(bbox_reshape1, (2, 3, 0, 1)), (-1, 4))

    order = np.argsort(conf_reshape2, 0)[-1]
    bbox_final = transform_bbox(bbox_reshape2[order], ANCHORS[order], img_width, img_height)

    return bbox_final

def feature_express_c(feature_map, img_width, img_height):
    lib = CDLL("./libgen_bbox.so")
    lib.preprocess.restype = c_void_p
    tensors = lib.preprocess()
    result = (c_float*4)()
    lib.feature_express(feature_map.ctypes.data_as(c_void_p), img_width, img_height, tensors, result)
    lib.postprocess(tensors)
    bbox = []*4
    bbox[0] = result[0]
    bbox[1] = result[1]
    bbox[2] = result[2]
    bbox[3] = result[3]
    return bbox

def test_with_file(filename, img_width, img_height):
    try:
        fo = open(filename)
        fstr = fo.read()
        tensor = parse_tensor_str(fstr)
    except IOError:
        print ("No such file: " + filename)
    finally:
        fo.close()
    # print(tensor)
    # print(np.shape(tensor))
    tensor = np.reshape(tensor, (144, 23, 40))
    start = time.time()
    bbox = feature_express(tensor, img_width, img_height)
    end = time.time()
    print ("predict:")
    print (bbox)
    print ("time of feature_express: %fms"%((end - start)*1000))
    return bbox

if __name__ == '__main__':
    filename = sys.argv[1]
    img_width = sys.argv[2]
    img_height = sys.argv[3]
    bbox = test_with_file(filename, int(img_width), int(img_height))
