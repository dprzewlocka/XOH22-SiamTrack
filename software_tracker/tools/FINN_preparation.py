import torch
import os
from siamfc import TrackerSiamFC

net_path = 'pretrained/AlexNetV5_I8H4A4O8/siamfc_QAlexNetV5_e39.pth'
tracker = TrackerSiamFC(net_path=net_path)

net_path = os.path.join('/home/vision/FINN/finn_07/finn/notebooks/siam_track/XOH', 'siamfc_test.pth')
torch.save(tracker.net.backbone.state_dict(), net_path)

print("That's all folks!")
