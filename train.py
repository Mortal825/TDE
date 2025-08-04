import os
from ultralytics import YOLO
import torch
import warnings

# 隐藏所有用户警告
warnings.filterwarnings("ignore", category=UserWarning)

os.environ['WANDB_DISABLED'] = 'true'

# baseline
model = YOLO("./config/snn_yolov8s.yaml")  #使用基线方式
# TCSA
model = YOLO("./config/TCSA_yolov8s.yaml")
# SDA
model = YOLO("./config/SDA_yolov8s.yaml")
# SE
model = YOLO("./config/SE_yolov8s.yaml")
# TDE(TCSA)
model = YOLO("./config/TDE(TCSA)_yolov8s.yaml")
# TDE(SDA)
model = YOLO("./config/TDE(SDA)_yolov8s.yaml")

# EvDET200K
model.train(data="./config/EvDET200K.yaml",device=[0,1],epochs=100,batch=12) 

# VOC
# model.train(data="./config/VOC.yaml",device=[0,1],epochs=100,batch=12) 

# # VOC2007
# model.train(data="./config/VOC07.yaml",device=[0,1],epochs=100,batch=12) 