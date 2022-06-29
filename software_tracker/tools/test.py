from __future__ import absolute_import

import os
from got10k.experiments import *

from siamfc import TrackerSiamFC


if __name__ == '__main__':
    net_path = 'pretrained/AlexNetV5_I8H4A4O8/siamfc_QAlexNetV5_e39.pth'
    # net_path = 'pretrained/siamfc_AlexNetV5_e50.pth'
    tracker = TrackerSiamFC(net_path=net_path, scale_num=3)

    root_dir = os.path.expanduser('~/data/VOT2015')
    e = ExperimentVOT(root_dir, read_image=False, version=2015, experiments=('unsupervised'))
    # e.run(tracker)
    e.report([tracker.name])

    # root_dir = os.path.expanduser('~/data/OTB')
    # e = ExperimentOTB(root_dir, version=2013)
    # e.run(tracker)
    # e.report([tracker.name])

    # setup experiment (validation subset)
    # experiment = ExperimentGOT10k(
    #     root_dir='/home/vision/data/GOT-10k',  # GOT-10k's root directory
    #     subset='test',  # 'train' | 'val' | 'test'
    #     result_dir='results',  # where to store tracking results
    #     report_dir='reports'  # where to store evaluation reports
    # )
    # experiment.run(tracker, visualize=False)
    # experiment.report([tracker.name])
