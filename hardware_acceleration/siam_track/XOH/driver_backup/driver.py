
# Copyright (c) 2020 Xilinx, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of Xilinx nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import numpy as np
import os
from finn.core.datatype import DataType
from driver_base import FINNExampleOverlay

# dictionary describing the I/O of the FINN-generated accelerator
io_shape_dict = {
    # FINN DataType for input and output tensors
    "idt" : [DataType['UINT8']],
    "odt" : [DataType['INT24']],
    # shapes for input and output tensors (NHWC layout)
    "ishape_normal" : [(1, 238, 238, 3)],
    "oshape_normal" : [(1, 22, 22, 128)],
    # folded / packed shapes below depend on idt/odt and input/output
    # PE/SIMD parallelization settings -- these are calculated by the
    # FINN compiler.
    "ishape_folded" : [(1, 238, 238, 1, 3)],
    "oshape_folded" : [(1, 22, 22, 16, 8)],
    "ishape_packed" : [(1, 238, 238, 1, 3)],
    "oshape_packed" : [(1, 22, 22, 16, 24)],
    "input_dma_name" : ['idma0'],
    "output_dma_name" : ['odma0'],
    "number_of_external_weights": 0,
    "num_inputs" : 1,
    "num_outputs" : 1,
}

# configuration & functions for SiamTrack (03.03.2022)
import cv2
import torch
import torch.nn as nn
import glob

cfg = {
        # basic parameters
        'out_scale': 0.001,
        'exemplar_sz': 110,
        'instance_sz': 238,
        'context': 0.5,
        # inference parameters
        'scale_num': 1,
        'scale_step': 1.0375,
        'scale_lr': 0.59,
        'scale_penalty': 0.9745,
        'window_influence': 0.176,
        'response_sz': 17,
        'response_up': 16,
        'total_stride': 8,
        # train parameters
        'epoch_num': 50,
        'batch_size': 8,
        'num_workers': 32,
        'initial_lr': 1e-2,
        'ultimate_lr': 1e-5,
        'weight_decay': 5e-4,
        'momentum': 0.9,
        'r_pos': 16,
        'r_neg': 0}

def read_image(img_file, cvt_code=cv2.COLOR_BGR2RGB):
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)
    return img

def save_image(img, boxes=None, box_fmt='ltwh', colors=None,
               thickness=3, fig_n=1, delay=1, visualize=True,
               cvt_code=cv2.COLOR_RGB2BGR):
    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)
    
    # resize img if necessary
    max_size = 960
    if max(img.shape[:2]) > max_size:
        scale = max_size / max(img.shape[:2])
        out_size = (
            int(img.shape[1] * scale),
            int(img.shape[0] * scale))
        img = cv2.resize(img, out_size)
        if boxes is not None:
            boxes = np.array(boxes, dtype=np.float32) * scale
    
    if boxes is not None:
        assert box_fmt in ['ltwh', 'ltrb']
        boxes = np.array(boxes, dtype=np.int32)
        if boxes.ndim == 1:
            boxes = np.expand_dims(boxes, axis=0)
        if box_fmt == 'ltrb':
            boxes[:, 2:] -= boxes[:, :2]
        
        # clip bounding boxes
        bound = np.array(img.shape[1::-1])[None, :]
        boxes[:, :2] = np.clip(boxes[:, :2], 0, bound)
        boxes[:, 2:] = np.clip(boxes[:, 2:], 0, bound - boxes[:, :2])
        
        if colors is None:
            colors = [
                (0, 0, 255),
                (0, 255, 0),
                (255, 0, 0),
                (0, 255, 255),
                (255, 0, 255),
                (255, 255, 0),
                (0, 0, 128),
                (0, 128, 0),
                (128, 0, 0),
                (0, 128, 128),
                (128, 0, 128),
                (128, 128, 0)]
        colors = np.array(colors, dtype=np.int32)
        if colors.ndim == 1:
            colors = np.expand_dims(colors, axis=0)
        
        for i, box in enumerate(boxes):
            color = colors[i % len(colors)]
            pt1 = (box[0], box[1])
            pt2 = (box[0] + box[2], box[1] + box[3])
            img = cv2.rectangle(img, pt1, pt2, color.tolist(), thickness)
    
    if visualize:
        winname = 'window_{}'.format(fig_n)
        cv2.imwrite('/home/xilinx/data/output/' + winname + '.jpg', img)
#         cv2.imshow(winname, img)
#         cv2.waitKey(delay)

    return img

def crop_and_resize(img, center, size, out_size,
                    border_type=cv2.BORDER_CONSTANT,
                    border_value=(0, 0, 0),
                    interp=cv2.INTER_LINEAR):
    # convert box to corners (0-indexed)
    size = round(size)
    corners = np.concatenate((
        np.round(center - (size - 1) / 2),
        np.round(center - (size - 1) / 2) + size))
    corners = np.round(corners).astype(int)

    # pad image if necessary
    pads = np.concatenate((
        -corners[:2], corners[2:] - img.shape[:2]))
    npad = max(0, int(pads.max()))
    if npad > 0:
        img = cv2.copyMakeBorder(
            img, npad, npad, npad, npad,
            border_type, value=border_value)

    # crop image patch
    corners = (corners + npad).astype(int)
    patch = img[corners[0]:corners[2], corners[1]:corners[3]]

    # resize to out_size
    patch = cv2.resize(patch, (out_size, out_size),
                       interpolation=interp)

    return patch

import time
import torch.nn.functional as F

class SiamFC(nn.Module):

    def __init__(self, out_scale=0.001):
        super(SiamFC, self).__init__()
        self.out_scale = out_scale
    
    def forward(self, z, x):
        return self._fast_xcorr(z, x) * self.out_scale
    
    def _fast_xcorr(self, z, x):
        # fast cross correlation
        nz = z.size(0)
        nx, c, h, w = x.size()
        x = x.view(-1, nz * c, h, w)
        out = F.conv2d(x, z, groups=nz)
        out = out.view(nx, -1, out.size(-2), out.size(-1))
        return out

class SiamTracker():
    
    def __init__(self, net=None):
        # These are values read from software part
        load_path = '/home/xilinx/data/Crossing/parameters/'
        self.center = np.load(load_path + 'center.npy')
        self.x_sz = np.load(load_path + 'x_sz.npy').item(0)
        self.avg_color = np.load(load_path + 'avg_color.npy')
        self.scale_factors = np.load(load_path + 'scale_factors.npy')
        self.device = 'cpu'
        self.head = SiamFC(cfg['out_scale'])
        self.kernel = torch.from_numpy(np.load(load_path + 'kernel.npy'))
        self.upscale_sz = np.load(load_path + 'upscale_sz.npy').item(0)
        self.hann_window = np.load(load_path + 'hann_window.npy')
        self.z_sz = np.load(load_path + 'z_sz.npy').item(0)
        self.target_sz = np.load(load_path + 'target_sz.npy')
        
        self.net = net

    def update(self, img):

        # search images
        x = [crop_and_resize(
            img, self.center, self.x_sz * f,
            out_size=cfg['instance_sz'],
            border_value=self.avg_color) for f in self.scale_factors]
        x = np.stack(x, axis=0)
        x = torch.from_numpy(x).to(
            self.device).permute(0, 3, 1, 2).float()

        # responses
        if self.net is not None:
            x = self.net(x)
        else:
            x = x.numpy()
            x = x[0, :, :, :][np.newaxis, :, :, :].transpose(0, 2, 3, 1)
            #input_dict = {iname: x.astype(np.float32)}
            #begin = time.time()
            x = accel.execute(x)
            #end = time.time()
            #print("FINN network execution: ", end - begin)
            if not isinstance(x, list):
                x = [x]
            #x = x[oname]
            x = np.array(x[0])
            #print("Shape of FINN output: ", x.shape)
            x = x.transpose(0, 3, 1, 2)
            x = x * 0.0002790141152217984 + np.load("/home/xilinx/data/Add_0_param0.npy")
            x = torch.from_numpy(x)
            
        #begin = time.time()
        responses = self.head(self.kernel, x)
        #end = time.time()
        #print("Cross correlation time: ", end - begin)
        responses = responses.squeeze(1).detach().numpy()

        # upsample responses and penalize scale changes
        #begin = time.time()
        responses = np.stack([cv2.resize(
            u, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC)
            for u in responses])
        responses[:cfg['scale_num'] // 2] *= cfg['scale_penalty']
        responses[cfg['scale_num'] // 2 + 1:] *= cfg['scale_penalty']
        #end = time.time()
        #print("Upsample time: ", end - begin)

        #begin = time.time()
        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location
        response = responses[scale_id]
        response -= response.min()
        response /= response.sum() + 1e-16
        response = (1 - cfg['window_influence']) * response + \
            cfg['window_influence'] * self.hann_window
        loc = np.unravel_index(response.argmax(), response.shape)

        # locate target center
        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
        disp_in_instance = disp_in_response * \
            cfg['total_stride'] / cfg['response_up']
        disp_in_image = disp_in_instance * self.x_sz * \
            self.scale_factors[scale_id] / cfg['instance_sz']
        self.center += disp_in_image

        #end = time.time()
        #print("Locating target time: ", end - begin)

        # update target size
        scale =  (1 - cfg['scale_lr']) * 1.0 + \
            cfg['scale_lr'] * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        # return 1-indexed and left-top based bounding box
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        return box
    
    def track(self, img_files, box, save_output=False):
        
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        times = np.zeros(frame_num)

        for f, img_file in enumerate(img_files):
            img = read_image(img_file)

            begin = time.time()
            if f == 0:
                continue
            else:
                boxes[f, :] = self.update(img)
            times[f] = time.time() - begin

            if boxes[f, 0] < 0 or boxes[f, 1] < 0 or boxes[f, 0] > img.shape[0] or boxes[f, 1] > img.shape[1]:
                print('Object lost. Aborting sequence tracking...')
                break

            if save_output:
                save_image(img, boxes[f, :], fig_n=f)
            
            if f > 1:
                break  # TODO for checking

        return boxes, times
# end of configuration & functions for SiamTrack

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Execute FINN-generated accelerator on numpy inputs, or run throughput test')
    parser.add_argument('--exec_mode', help='Please select functional verification ("execute") or throughput test ("throughput_test")', default="execute")
    parser.add_argument('--platform', help='Target platform: zynq-iodma alveo', default="zynq-iodma")
    parser.add_argument('--batchsize', help='number of samples for inference', type=int, default=1)
    parser.add_argument('--bitfile', help='name of bitfile (i.e. "resizer.bit")', default="resizer.bit")
    parser.add_argument('--inputfile', help='name(s) of input npy file(s) (i.e. "input.npy")', nargs="*", type=str, default=["input.npy"])
    parser.add_argument('--outputfile', help='name(s) of output npy file(s) (i.e. "output.npy")', nargs="*", type=str, default=["output.npy"])
    parser.add_argument('--runtime_weight_dir', help='path to folder containing runtime-writable .dat weights', default="runtime_weights/")
    # parse arguments
    args = parser.parse_args()
    exec_mode = args.exec_mode
    platform = args.platform
    batch_size = args.batchsize
    bitfile = args.bitfile
    inputfile = args.inputfile
    outputfile = args.outputfile
    runtime_weight_dir = args.runtime_weight_dir

    # instantiate FINN accelerator driver and pass batchsize and bitfile
    accel = FINNExampleOverlay(
        bitfile_name = bitfile, platform = platform,
        io_shape_dict = io_shape_dict, batch_size = batch_size,
        runtime_weight_dir = runtime_weight_dir
    )

    # for the remote execution the data from the input npy file has to be loaded,
    # packed and copied to the PYNQ buffer
    if exec_mode == "execute":
        # load desired input .npy file(s)
        ibuf_normal = []
        for ifn in inputfile:
            ibuf_normal.append(np.load(ifn))
        obuf_normal = accel.execute(ibuf_normal)
        if not isinstance(obuf_normal, list):
            obuf_normal = [obuf_normal]
        for o, obuf in enumerate(obuf_normal):
            np.save(outputfile[o], obuf)
    elif exec_mode == "throughput_test":
        # remove old metrics file
        try:
            os.remove("nw_metrics.txt")
        except FileNotFoundError:
            pass
        res = accel.throughput_test()
        file = open("nw_metrics.txt", "w")
        file.write(str(res))
        file.close()
        print("Results written to nw_metrics.txt")
    elif exec_mode == "siamese_tracking":
        seq_dir = os.path.expanduser('/home/xilinx/data/Crossing/')
        img_files = sorted(glob.glob(seq_dir + 'img/*.jpg'))
        anno = np.loadtxt(seq_dir + 'groundtruth_rect.txt')

        tracker = SiamTracker()

        boxes, times = tracker.track(img_files, anno[0], save_output=True)
        print("All frames' times: ", times[:5])
    else:
        raise Exception("Exec mode has to be set to remote_pynq or throughput_test")
