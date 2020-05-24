# Author: Deepak Pathak (c) 2016

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
import numpy as np
from PIL import Image
import time
import argparse
import pyflow
import cv2
import os

def save_img(flow, flow_index):
    hsv = np.zeros(im1.shape, dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite('output/img/%s-%s.png' % (flow_index, flow_index+1), rgb)

def get_img_path(dir_path, frame_index):
    return '%s/%s.jpg' % (dir_path, str(frame_index).rjust(8, '0'))

images_path = '/home/k-takashima/workspace/research/dataset/flow_data/1_1_original'

parser = argparse.ArgumentParser(
    description='Demo for python wrapper of Coarse2Fine Optical Flow')
parser.add_argument(
    '-viz', dest='viz', action='store_true',
    help='Visualize (i.e. save) output of flow.')
args = parser.parse_args()


# Flow Options:
alpha = 0.012
ratio = 0.75
minWidth = 20
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30
colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

flow_all = np.ndarray(shape=(20, 480, 620, 2), dtype='float') # stacked flow data

frame_index = 1 # image 00000001.jpg
while os.path.isfile(get_img_path(images_path, frame_index+1)):
    im1 = np.array(Image.open(get_img_path(images_path, frame_index)))
    im2 = np.array(Image.open(get_img_path(images_path, frame_index+1)))
    im1 = im1.astype(float) / 255.
    im2 = im2.astype(float) / 255.
    s = time.time()
    u, v, im2W = pyflow.coarse2fine_flow(
        im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
        nSORIterations, colType)
    e = time.time()
    print('Time Taken: %.2f seconds for image of size (%d, %d, %d)' % (
        e - s, im1.shape[0], im1.shape[1], im1.shape[2]))
    # flow data dx=u, dy=v
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    save_img(flow, frame_index)
    np.append(flow_all, flow)
    print('append flow %s-%s' % (str(frame_index).rjust(8, '0'), str(frame_index+1).rjust(8, '0')))
    frame_index += 1

print('##### complete #####')
np.save('output/outFlow.npy', flow_all)
print('Flow shape: %s' % (flow_all.shape,))

