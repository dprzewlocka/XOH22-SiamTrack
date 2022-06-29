from __future__ import absolute_import

import os
import glob
import numpy as np

from siamfc import TrackerSiamFC


if __name__ == '__main__':
    seq_dir = os.path.expanduser('~/data/OTB/Crossing/')
    img_files = sorted(glob.glob(seq_dir + 'img/*.jpg'))
    anno = np.loadtxt(seq_dir + 'groundtruth_rect.txt')
    
    net_path = 'pretrained/AlexNetV5_I8H4A4O8/siamfc_QAlexNetV5_e39.pth'
    # net_path = '/home/vision/FINN/finn_07/finn/notebooks/siam_track/XOH/siamfc.pth'
    tracker = TrackerSiamFC(net_path=net_path, scale_num=1)
    tracker.track(img_files, anno[0], visualize=True)
