import os
import glob
import numpy as np
import time
from util import cal_rel_pose
from torchvision import transforms
from PIL import Image
import torch

# calculate relative poses in a format of [theta_x (roll), theta_y (pitch), theta_z (yaw), x, y, z]
# save as .npy file
def create_pose_data(pose_dir):

    info = {'00': [0, 4540], '01': [0, 1100], '02': [0, 4660], '03': [0, 800], '04': [0, 270], '05': [0, 2760], '06': [0, 1100], '07': [0, 1100], '08': [1100, 5170], '09': [0, 1590], '10': [0, 1200]}
    start_t = time.time()
    for video in info.keys():
        fn = '{}/{}.txt'.format(pose_dir, video)
        print('Transforming {}...'.format(fn))
        with open(fn) as f:
            lines = [line.split('\n')[0] for line in f.readlines()]
            poses = []
            for i in range(len(lines)):
                values = [float(value) for value in lines[i].split(' ')]
                if i > 0:
                    values_pre = [float(value) for value in lines[i-1].split(' ')]
                    poses.append(cal_rel_pose(values_pre, values))                     
            poses = np.array(poses)
            base_fn = os.path.splitext(fn)[0]
            np.save(base_fn+'.npy', poses)
            print('Video {}: shape={}'.format(video, poses.shape))
    print('elapsed time = {}'.format(time.time()-start_t))


if __name__ == '__main__':
    pose_dir = '/home/mingyuy/DeepVO/dataset/poses'
    create_pose_data(pose_dir)