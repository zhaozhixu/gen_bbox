import numpy as np
import sys
import math
import os
import time
from ctypes import *

W_VALID = 20
H_VALID = 12
IMG_H = 360
IMG_W = 640
ANCHORS_PER_GRID = 9
ANCHOR_SHAPE = [ 229., 137., 48., 71., 289., 245.,
                 185., 134., 85., 142., 31., 41.,
                 197., 191., 237., 206., 63., 108.]

def prepare_anchors(anchor_shape, input_w, input_h, convout_w, convout_h,
                    anchors_per_grid):
    center_x = np.zeros(convout_w, dtype=np.float32)
    center_y = np.zeros(convout_h, dtype=np.float32)
    anchors = np.zeros(convout_h*convout_w*anchors_per_grid*4, dtype=np.float32)

    for i in range(convout_w):
        center_x[i] = (i + 0.5) * input_w / (convout_w + 0.0)
    for i in range(convout_h):
        center_y[i] = (i + 0.5) * input_h / (convout_h + 0.0)

    h_vol = convout_w * anchors_per_grid * 4
    w_vol = anchors_per_grid * 4
    b_vol = 4
    for i in range(convout_h):
        for j in range(convout_w):
            for k in range(anchors_per_grid):
                anchors[i*h_vol+j*w_vol+k*b_vol] = center_x[j]
                anchors[i*h_vol+j*w_vol+k*b_vol+1] = center_y[i]
                anchors[i*h_vol+j*w_vol+k*b_vol+2] = anchor_shape[k*2]
                anchors[i*h_vol+j*w_vol+k*b_vol+3] = anchor_shape[k*2+1]

    return anchors

anchors = None
lib = None
lib_data = None
result = None
bbox = None
def init():
    global anchors, lib, lib_data, result, bbox
    anchors = prepare_anchors(ANCHOR_SHAPE, IMG_W, IMG_H, W_VALID,
                              H_VALID, ANCHORS_PER_GRID)
    lib = CDLL("./libgen_bbox_dpu.so")
    lib_data = lib.gbd_preprocess()
    result = (c_float*4)()

def detect(feature):
    global anchors, lib, lib_data, result, bbox
    lib.gbd_getbbox(lib_data, feature.ctypes.data_as(c_void_p),
                    anchors.ctypes.data_as(c_void_p),
                    result)
    return [result[0], result[1], result[2], result[3]]

def cleanup():
    global lib_data
    lib.gbd_postprocess(lib_data)

init()
filename = sys.argv[1]
feature = np.loadtxt(filename, delimiter='\t').astype(np.int8)
# feature = np.loadtxt("test.txt", delimiter='\t').astype(np.int8)
start=time.time()
bbox = detect(feature)
end = time.time()
cleanup()

# xmin, xmax, ymin, ymax
print (bbox)
print ("time: %.3fms"%((end - start)*1000))
