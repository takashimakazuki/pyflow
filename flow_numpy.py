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
from glob import glob
from pathlib import Path


def save_img(flow, flow_index, shape):
    hsv = np.zeros(shape, dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    print(rgb)
    cv2.imwrite('output/flow_from_npy/%s-%s.png' % (flow_index, flow_index+1), rgb)



# Flow Options:
alpha = 0.012
ratio = 0.75
minWidth = 20
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30
colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))


def gen_flow(input_dir, out_dir):
    flow_ary = [] # flow data
    frames = np.load(os.path.join(input_dir, 'frames.npy'))
    for frame_index in range(frames.shape[0]-1):
        im1 = frames[frame_index]
        im2 = frames[frame_index+1]
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
        #save_img(flow, frame_index, im1.shape)
        flow_ary.append(flow)
        print('append flow %s-%s' % (str(frame_index).rjust(8, '0'), str(frame_index+1).rjust(8, '0')))
        frame_index += 1

    print('##### complete #####')
    flow_ndarray = np.concatenate([arr[np.newaxis] for arr in flow_ary])
    print('output: %s', os.path.join(out_dir, 'frames.npy'))
    np.save(os.path.join(out_dir, 'frames.npy'), flow_ndarray)


# NOTE: path should be terminated with "/"
data_path = '/home/k-takashima/workspace/research/dataset/data_c3d/'
output_path = '/home/k-takashima/workspace/research/dataset/data_flow'

os.mkdir(output_path)
for camera_dir in glob(os.path.join(data_path, '*')):
    camera = os.path.basename(camera_dir)
    os.mkdir(os.path.join(output_path, camera))
    for waza_dir in glob(os.path.join(camera_dir, '*')):
        waza = os.path.basename(waza_dir)
        os.mkdir(os.path.join(output_path, camera, waza))
        for human_dir in glob(os.path.join(waza_dir, '*')):
            human = os.path.basename(human_dir)
            os.mkdir(os.path.join(output_path, camera, waza, human))
            out = os.path.join(output_path, camera, waza, human)
            gen_flow(human_dir, out)


