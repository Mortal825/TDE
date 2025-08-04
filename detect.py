# -*- coding: utf-8 -*-
import os
from PIL import Image
from ultralytics import YOLO
import os
from ultralytics import YOLO
import torch.nn as nn
import torch
# from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode
from ultralytics.nn.modules.yolo_spikformer import MultiSpike4,MultiSpike2,MultiSpike1,mem_update,Time_Decoder
from ultralytics.nn.modules.Attention import TimeAttention
from collections import Counter
import matplotlib.pyplot as plt
folder_path = './result'
os.makedirs(folder_path, exist_ok=True)
model = YOLO("/home/haichao/luofan/SpikeYOLO/runs/detect/EVDET_TDE(SAM)/weights/best.pt").to("cuda:2")
results = model(['/home/haichao/luofan/SpikeYOLO/figure/000516.jpg'])
#新建一个文件夹保存图片
import os



for i, r in enumerate(results):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    im_rgb.show()
    im_rgb.save(os.path.join(folder_path,'result.jpg'))  # save to disk



















































