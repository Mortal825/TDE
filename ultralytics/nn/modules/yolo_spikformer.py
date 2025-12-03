# from visualizer import get_local
import torch
import torchinfo
import torch.nn as nn
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from einops.layers.torch import Rearrange
from spikingjelly.activation_based import neuron

import torch.nn.functional as F
from functools import partial
from einops import rearrange
import warnings
# from visualizer import get_local
from ultralytics.nn.modules.seg import MulfeatSeg
from ultralytics.utils.tal import TORCH_1_10, dist2bbox, make_anchors
import math
from ultralytics.nn.modules.Attention import TCSA,Spike_TCSA
import matplotlib.pyplot as plt
import numpy as np
# __all__ = ('MS_GetT','MS_CancelT', 'MS_ConvBlock','MS_Block','MS_DownSampling',
#            'MS_StandardConv','SpikeSPPF','SpikeConv','MS_Concat','SpikeDetect'
#            ,'Ann_ConvBlock','Ann_DownSampling','Ann_StandardConv','Ann_SPPF','MS_C2f',
#            'Conv_1','BasicBlock_1','BasicBlock_2','Concat_res2','Sample','MS_FullConvBlock','MS_ConvBlock_resnet50','MS_AllConvBlock','MS_ConvBlock_res2net')

decay = 0.25  # 0.25 # decay constants


alpha = 0.0
num_layer = 0
from spikingjelly.activation_based import neuron

## 用于网络中正常的脉冲更新
class mem_update(nn.Module):
    def __init__(self, act=False):
        super(mem_update, self).__init__()
        # self.actFun= torch.nn.LeakyReLU(0.2, inplace=False)
        self.act = act
        self.qtrick = MultiSpike1()  # change the max value

    def forward(self, x):
        global alpha,num_layer
        spike = torch.zeros_like(x[0]).to(x.device)
        output = torch.zeros_like(x)
        mem_old = 0
        time_window = x.shape[0]
        for i in range(time_window):
            if i >= 1:
                # mem = (mem_old - spike.detach()) * decay + x[i]   ## 这是Baseline使用的,再试一次这个是不是更好，还是没有影响。
                mem = (mem_old - spike.detach()*0.5) * decay + x[i]   #spike被detach了，不会产生梯度这里
            else:
                mem = x[i]
            spike = self.qtrick(mem)
            mem_old = mem.clone()
            output[i] = spike
            
        return output

## SDA
from torch.cuda.amp import autocast

class nomem_update(nn.Module):
    def __init__(self, ratio=0.9, detach=True):
        super(nomem_update, self).__init__()
        self.ratio = ratio
        self.detach = detach

    def forward(self, x):
        # 使用 AMP 自动管理精度
        with autocast(enabled=True):  # 根据需要使用 AMP
            flat = x.view(-1)
            k = int(flat.numel() * self.ratio)

            if k == 0:
                mask = torch.zeros_like(x)
            else:
                threshold = torch.topk(flat, k).values[-1]
                mask = (x >= threshold).to(x.dtype)  # 使用 x 的 dtype，保持一致
            if self.detach:
                out = mask.detach()
            else:
                out = x + (mask - x).detach()  # 使用 STE
        # print(out)
        return out


class getmem_update(nn.Module):
    def __init__(self, act=False):
        super(getmem_update, self).__init__()
        # self.actFun= torch.nn.LeakyReLU(0.2, inplace=False)
        self.act = act
        self.qtrick = MultiSpike1()  # change the max value

    def forward(self, x):
        spike = self.qtrick(x)
        return spike,x
    

class MultiSpike8(nn.Module):  # 直接调用实例化的quant6无法实现深拷贝。解决方案是像下面这样用嵌套的类
    class quant8(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.round(torch.clamp(input, min=0, max=8))

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            #             print("grad_input:",grad_input)
            grad_input[input < 0] = 0
            grad_input[input > 8] = 0
            return grad_input

    def forward(self, x):
#         print(self.quant8.apply(x))
        return self.quant8.apply(x)

class MultiSpike4(nn.Module):

    class quant4(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            # print("当前用的4")
            ctx.save_for_backward(input)
            return torch.round(torch.clamp(input, min=0, max=4))

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            #             print("grad_input:",grad_input)
            grad_input[input < 0] = 0
            grad_input[input > 4] = 0
            return grad_input

    def forward(self, x):
        return self.quant4.apply(x)

class MultiSpike2(nn.Module):  # 直接调用实例化的quant6无法实现深拷贝。解决方案是像下面这样用嵌套的类

    class quant2(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            # print("当前用的2")
            ctx.save_for_backward(input)
            return torch.round(torch.clamp(input, min=0, max=2))

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            #             print("grad_input:",grad_input)
            grad_input[input < 0] = 0
            grad_input[input > 2] = 0
            return grad_input

    def forward(self, x):
        return self.quant2.apply(x)

class MultiSpike1(nn.Module):

    class quant1(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            # print("当前用的1")
            ctx.save_for_backward(input)
            return torch.round(torch.clamp(input, min=0, max=1))
            # return torch.floor(torch.clamp(input, min=0, max=1))

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            #             print("grad_input:",grad_input)
            ## 量化操作输出在这些区域不会因为输入而变化，因此梯度需要置为0
            grad_input[input < 0] = 0
            grad_input[input > 1] = 0
            return grad_input

    def forward(self, x):
        return self.quant1.apply(x)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p



@torch.jit.script
def jit_mul(x, y):
    return x.mul(y)

@torch.jit.script
def jit_sum(x):
    return x.sum(dim=[-1, -2], keepdim=True)

class SpikeDFL(nn.Module):
    """    
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)  #[0,1,2,...,15]
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1)) #这里不是脉冲驱动的，但是是整数乘法
        self.c1 = c1  #本质上就是个加权和。输入是每个格子的概率(小数)，权重是每个格子的位置(整数)
        self.lif = mem_update()


    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)  # 原版

class SpikeDetect(nn.Module):
    """YOLOv8 Detect head for detection models."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(SpikeConv(x, c2, 3), SpikeConv(c2, c2, 3), SpikeConvWithoutBN(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(SpikeConv(x, c3, 3), SpikeConv(c3, c3, 3), SpikeConvWithoutBN(c3, self.nc, 1)) for x in ch)
        self.dfl = SpikeDFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        shape = x[0].mean(0).shape  # BCHW  推理：[1，2，64，32，84]  这里必须mean0，否则推理时用到shape会导致报错
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 2)
            x[i] = x[i].mean(0)  #[2，144，32，684]  #这个地方有时候全是1.之后debug看看
            # print(f"x[i] = {x[i].shape}")
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1) #box: [B,reg_max * 4,anchors]
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

        if self.export and self.format in ('tflite', 'edgetpu'):
            # Normalize xywh with image size to mitigate quantization error of TFLite integer models as done in YOLOv5:
            # https://github.com/ultralytics/yolov5/blob/0c8de3fca4a702f8ff5c435e67f378d1fce70243/models/tf.py#L307-L309
            # See this PR for details: https://github.com/ultralytics/ultralytics/pull/1695
            img_h = shape[2] * self.stride[0]
            img_w = shape[3] * self.stride[0]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=dbox.device).reshape(1, 4, 1)
            dbox /= img_size

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].conv.bias.data[:] = 1.0  # box
            b[-1].conv.bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

## 时间步解码

class Time_Decoder(nn.Module):
    def __init__(self, T):
        super(Time_Decoder, self).__init__()
        # 定义全连接层，将 T 映射到 1
        self.fc = nn.Linear(T, 1)
        # # 初始化权重和偏置为0.25
        nn.init.constant_(self.fc.weight, 0.25)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        # 输入形状: [T, 2, 144, 80, 80]
        T, B, C, H, W = x.shape

        # 将 T 维度展平到最后
        x = x.permute(1, 2, 3, 4, 0)  # 调整形状为 [2, 144, 80, 80, T]
        x = x.flatten(0, -2)          # 调整形状为 [(2*144*80*80), T]

        # 应用全连接层
        x = self.fc(x)  # 输出形状为 [(2*144*80*80), 1]

        # 恢复形状
        x = x.view(B, C, H, W)    # 调整形状为 [2, 144, 80, 80, 1]

        return x


class AuxSeg(nn.Module):
    def __init__(self,in_channels, height, width):
        super(AuxSeg, self).__init__()
        self.net = MulfeatSeg(in_channels, height, width)
    def forward(self, x):
        x =  self.net(x)
        # print(f"x = {x.shape}")
        return x

class SpikeDetect_TD(nn.Module):
    """YOLOv8 Detect head for detection models."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, T = 2 ,ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(SpikeConv(x, c2, 3), SpikeConv(c2, c2, 3), SpikeConvWithoutBN(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(SpikeConv(x, c3, 3), SpikeConv(c3, c3, 3), SpikeConvWithoutBN(c3, self.nc, 1)) for x in ch)
        self.dfl = SpikeDFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        self.td = Time_Decoder(T = T)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        shape = x[0].mean(0).shape  # BCHW  推理：[1，2，64，32，84]  这里必须mean0，否则推理时用到shape会导致报错
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 2)
            x[i] = self.td(x[i])
            # x[i] = x[i].mean(0)  #[2，144，32，684]  #这个地方有时候全是1.之后debug看看
        if self.training:
            # for k in range(len(x)):
            #     print(f"x{k} = {x[k].shape}")
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1) #box: [B,reg_max * 4,anchors]
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

        if self.export and self.format in ('tflite', 'edgetpu'):
            # Normalize xywh with image size to mitigate quantization error of TFLite integer models as done in YOLOv5:
            # https://github.com/ultralytics/yolov5/blob/0c8de3fca4a702f8ff5c435e67f378d1fce70243/models/tf.py#L307-L309
            # See this PR for details: https://github.com/ultralytics/ultralytics/pull/1695
            img_h = shape[2] * self.stride[0]
            img_w = shape[3] * self.stride[0]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=dbox.device).reshape(1, 4, 1)
            dbox /= img_size

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].conv.bias.data[:] = 1.0  # box
            b[-1].conv.bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
  

class Aux_SpikeDetect_TD(nn.Module):
    """YOLOv8 Detect head for detection models."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, in_channels, height, width, nc=80, T = 2 ,ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(SpikeConv(x, c2, 3), SpikeConv(c2, c2, 3), SpikeConvWithoutBN(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(SpikeConv(x, c3, 3), SpikeConv(c3, c3, 3), SpikeConvWithoutBN(c3, self.nc, 1)) for x in ch)
        self.dfl = SpikeDFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        self.td = Time_Decoder(T = T)
        self.net = MulfeatSeg(in_channels, height, width)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        aux = x.copy()
        shape = x[0].mean(0).shape  # BCHW  推理：[1，2，64，32，84]  这里必须mean0，否则推理时用到shape会导致报错
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 2)
            x[i] = self.td(x[i])
            # x[i] = x[i].mean(0)  #[2，144，32，684]  #这个地方有时候全是1.之后debug看看
        if self.training:
            aux = self.net(aux)
            x.append(aux)
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape


        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1) #box: [B,reg_max * 4,anchors]
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

        if self.export and self.format in ('tflite', 'edgetpu'):
            # Normalize xywh with image size to mitigate quantization error of TFLite integer models as done in YOLOv5:
            # https://github.com/ultralytics/yolov5/blob/0c8de3fca4a702f8ff5c435e67f378d1fce70243/models/tf.py#L307-L309
            # See this PR for details: https://github.com/ultralytics/ultralytics/pull/1695
            img_h = shape[2] * self.stride[0]
            img_w = shape[3] * self.stride[0]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=dbox.device).reshape(1, 4, 1)
            dbox /= img_size

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].conv.bias.data[:] = 1.0  # box
            b[-1].conv.bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
  
class BNAndPadLayer(nn.Module):
    def __init__(
            self,
            pad_pixels,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
    ):
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm2d(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = (
                        self.bn.bias.detach()
                        - self.bn.running_mean
                        * self.bn.weight.detach()
                        / torch.sqrt(self.bn.running_var + self.bn.eps)
                )
            else:
                pad_values = -self.bn.running_mean / torch.sqrt(
                    self.bn.running_var + self.bn.eps
                )
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0: self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels:, :] = pad_values
            output[:, :, :, 0: self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels:] = pad_values
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps

class RepConv(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size = 3,
            bias=False,
            group = 1
    ):
        super().__init__()
        padding = int((kernel_size-1)/2)
        # hidden_channel = in_channel
        conv1x1 = nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=False, groups=group)
        bn = BNAndPadLayer(pad_pixels=padding, num_features=in_channel)
        conv3x3 = nn.Sequential(
            # mem_update(), #11111
            nn.Conv2d(in_channel, in_channel, kernel_size, 1,0, groups=in_channel, bias=False),  #这里也是分组卷积
            # mem_update(),  #11111
            nn.Conv2d(in_channel, out_channel, 1,  1,0, groups=group, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        self.body = nn.Sequential(conv1x1, bn, conv3x3)

    def forward(self, x):
        return self.body(x)

    
class SepRepConv(nn.Module): #放在Sepconv最后一个1*1卷积，采用3*3分组+1*1降维的方式实现，能提0.5个点。之后可以试试改成1*1降维和3*3分组
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size = 3,
            bias=False,
            group = 1
    ):
        super().__init__()
        padding = int((kernel_size-1)/2)
        # hidden_channel = in_channel
#         conv1x1 = nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=False, groups=group)
        bn = BNAndPadLayer(pad_pixels=padding, num_features=in_channel)
        conv3x3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, 1,0, groups=group, bias=False),  #这里也是分组卷积
            # mem_update(), #11111
            nn.Conv2d(out_channel, out_channel, kernel_size,  1,0, groups=out_channel, bias=False),
        )
        self.body = nn.Sequential(bn, conv3x3)

    def forward(self, x):
        return self.body(x)


class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(self,
                 dim,
                 expansion_ratio=2,
                 act2_layer=nn.Identity,
                 bias=False,
                 kernel_size=3,  #7,3
                 padding=1):
        super().__init__()
        padding = int((kernel_size -1)/2)
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias)
        self.dwconv2 = nn.Conv2d(
            med_channels, med_channels, kernel_size=kernel_size, #7*7
            padding=padding, groups=med_channels, bias=bias)  # depthwise conv
#         self.pwconv3 = nn.Conv2d(med_channels, dim, kernel_size=1, stride=1, bias=bias,groups=1)
        self.pwconv3=SepRepConv(med_channels, dim)  #这里将sepconv最后一个卷积替换为重参数化卷积  大概提0.5个点，可以保留

        self.bn1 = nn.BatchNorm2d(med_channels)
        self.bn2 = nn.BatchNorm2d(med_channels)
        self.bn3 = nn.BatchNorm2d(dim)


        self.lif1 = mem_update()
        self.lif2 = mem_update()
        self.lif3 = mem_update()

    def forward(self, x):
        T, B, C, H, W = x.shape
        # print("x.shape:",x.shape)
        # for i in range(T):
        #     print(f"第{i}个时间步x的均值",x[i].mean())
        x = self.lif1(x) #x1_lif:0.2328  x2_lif:0.0493  这里x2的均值偏小，因此其经过bn和lif后也偏小，发放率比较低；而x1均值偏大，因此发放率也高
        
        x = self.bn1(self.pwconv1(x.flatten(0, 1))).reshape(T, B, -1, H, W)  # flatten：从第0维开始，展开到第一维
        x = self.lif2(x)
        x = self.bn2(self.dwconv2(x.flatten(0, 1))).reshape(T, B, -1, H, W)
        x = self.lif3(x)
        x = self.bn3(self.pwconv3(x.flatten(0, 1))).reshape(T, B, -1, H, W)
        return x

def generate_mask_upper_triangle_mat(head_num: int = 1, T: int = 1, N: int = 1):
    mask_mat_raw = torch.ones(T * N, T * N)
    for t in range(T - 1):
        mask_mat_raw[t * N:(t + 1) * N, (t + 1) * N:] = 0.
    mask_mat = mask_mat_raw.unsqueeze(0).repeat(head_num, 1, 1)
    return mask_mat

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, forward_drop=0., tau=0.5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = nn.Conv3d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_drop = nn.Dropout(p=forward_drop)
        self.fc1_bn = nn.BatchNorm3d(hidden_features)
        self.fc1_lif = mem_update()

        self.fc2_conv = nn.Conv3d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_drop = nn.Dropout(p=forward_drop)
        self.fc2_bn = nn.BatchNorm3d(out_features)
        self.fc2_lif = mem_update()

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        x = x.permute(1, 2, 0, 3, 4).contiguous()
        x = self.fc1_conv(x)
        x = self.fc1_drop(x)
        x = self.fc1_bn(x).permute(2, 0, 1, 3, 4).contiguous()
        x = self.fc1_lif(x)

        x = x.permute(1, 2, 0, 3, 4).contiguous()
        x = self.fc2_conv(x)
        x = self.fc2_drop(x)
        x = self.fc2_bn(x).permute(2, 0, 1, 3, 4).contiguous()
        x = self.fc2_lif(x)
        # print(f"mlp_x = {x.shape}")
        return x


class STSA(nn.Module):
    def __init__(self, dim, num_heads=1, attn_drop=0., proj_drop=0., T=0, H=0, W=0, tau=0.5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = [T, H, W]
        print(f"window_size = {self.window_size}")
        print(f"T = {T}")
        mask_mat = generate_mask_upper_triangle_mat(num_heads, T, H * W)
        self.register_buffer('mask_mat', mask_mat)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1),
                        num_heads))
        print(f'bias_table = {self.relative_position_bias_table.shape}')

        trunc_normal_(self.relative_position_bias_table, std=.02)

        # get pair-wise relative position index for each token of a DVS sample
        coordinates_t = torch.arange(self.window_size[0])
        coordinates_h = torch.arange(self.window_size[1])
        coordinates_w = torch.arange(self.window_size[2])
        coordinates = torch.stack(torch.meshgrid(coordinates_t, coordinates_h, coordinates_w))
        coordinates_flatten = torch.flatten(coordinates, 1)
        relative_coordinates = coordinates_flatten[:, :, None] - coordinates_flatten[:, None, :]
        relative_coordinates = relative_coordinates.permute(1, 2, 0).contiguous()
        relative_coordinates[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coordinates[:, :, 1] += self.window_size[1] - 1
        relative_coordinates[:, :, 2] += self.window_size[2] - 1

        # Avoid having the same index number in different locations
        relative_coordinates[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coordinates[:, :, 1] *= (2 * self.window_size[2] - 1)

        # Generate the final location index
        relative_position_index = relative_coordinates.sum(-1)
        print(f"relative_position_index = {relative_position_index.shape}")
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv_conv = nn.Conv1d(dim, dim * 3, kernel_size=1, stride=1, bias=False)
        self.qkv_bn = nn.BatchNorm1d(dim * 3)
        self.qkv_lif = mem_update()

        self.to_qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.to_qkv_bn = nn.BatchNorm1d(dim * 3)
        self.to_qkv_lif = mem_update()

        self.attn_lif = mem_update()
        self.attn_drop = nn.Dropout(p=attn_drop)

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = mem_update()
        self.proj_drop = nn.Dropout(p=proj_drop)

    def forward(self, x):
        T, B, C, H, W = x.shape
        # print(f"STSA T,B,C,H,W = {x.shape}")
        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.reshape(B, C, -1).contiguous()  # B, C, T * N
        qkv_out = self.qkv_conv(x_for_qkv)
        qkv_out = self.qkv_bn(qkv_out).reshape(B, C * 3, T, N).permute(2, 0, 1, 3).contiguous()
        qkv_out = self.qkv_lif(qkv_out).permute(1, 0, 3, 2).chunk(3, dim=-1)
        # print(f"h = {self.num_heads}")
        q, k, v = map(lambda z: rearrange(z, 'b t n (h d) -> b h (t n) d', h=self.num_heads), qkv_out)
        # print(f"q = {q.shape}")
        # compute the STRPB
        # print(f"relative_position_index = {self.relative_position_index.shape}")
        # print(f"self.relative_position_bias_table = {self.relative_position_bias_table.shape}")
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:T * N, :T * N].reshape(-1)].reshape(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1)

        attn = (q @ k.transpose(-2, -1))  # B, head_num, token_num, token_num
        # print(f"attn = {attn.shape}")
        # print(f"relative_position_bias = {relative_position_bias.shape}")
        # add STRPB
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = attn * self.mask_mat.unsqueeze(0)

        x = (attn @ v) * 0.125

        x = x.permute(0, 2, 1, 3).reshape(B, T * N, C).reshape(B, T, N, C).permute(1, 0, 3, 2).contiguous()
        x = self.attn_lif(x)  # T, B, C, N
        x = x.permute(1, 2, 0, 3).contiguous()  # B, C, T, N
        x = x.reshape(B, C, -1).contiguous()  # B, C, T*N
        x = self.proj_conv(x)
        x = self.proj_lif(self.proj_bn(x).reshape(B, C, T, N).permute(2, 0, 1, 3).reshape(T, B, C, H, W).contiguous())

        # print(f"x = {x.shape}")

        return x


class encoder_block(nn.Module):
    def __init__(self, dim, num_heads, T, H, W, mlp_ratio, proj_drop=0., attn_drop=0.,
                 forward_drop=0.3,
                 drop_path=0., tau=0.5):
        super().__init__()
        self.attn = STSA(dim, num_heads=num_heads,
                         attn_drop=attn_drop, proj_drop=proj_drop,
                         T=T, H=H, W=W, tau=tau)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, forward_drop=forward_drop, tau=tau)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class MS_ConvBlock(nn.Module):
    def __init__(self, input_dim, mlp_ratio=4.,sep_kernel_size = 7 ,timestep = 4,full=False):  # in_channels(out_channels), 内部扩张比例
        super().__init__()

        self.full =full
        self.Conv = SepConv(dim=input_dim,kernel_size= sep_kernel_size)  #内部扩张2倍
        self.mlp_ratio = mlp_ratio
        self.Tsteps = timestep
        self.lif1 = mem_update()
        self.lif2 = mem_update()

        self.conv1 = RepConv(input_dim, int(input_dim * mlp_ratio)) #137以外的模型，在第一个block不做分组

        self.bn1 = nn.BatchNorm2d(int(input_dim * mlp_ratio))  # 这里可以进行改进

        self.conv2 = RepConv(int(input_dim * mlp_ratio), input_dim)
        self.bn2 = nn.BatchNorm2d(input_dim)  # 这里可以进行改进

    # @get_local('x_feat')
    def forward(self, x):
        T, B, C, H, W = x.shape
        # print(f"MS_ConvBlockinput = {x.shape}")
        x = self.Conv(x) + x  #sepconv  pw+dw+pw

        x_feat = x

        x = self.bn1(self.conv1(self.lif1(x).flatten(0, 1))).reshape(T, B, int(self.mlp_ratio * C), H, W)
            #repconv，对应conv_mixer，包含1*1,3*3,1*1三个卷积，等价于一个3*3卷积
        x = self.bn2(self.conv2(self.lif2(x).flatten(0, 1))).reshape(T, B, C, H, W)

        x = x_feat + x
        return x

## 放在最外面
class att_MS_ConvBlock(nn.Module):
    def __init__(self, input_dim, mlp_ratio=4.,sep_kernel_size = 7 ,att = 0,timestep = 4,full=False):  # in_channels(out_channels), 内部扩张比例
        super().__init__()

        self.full =full
        self.Conv = SepConv(dim=input_dim,kernel_size= sep_kernel_size)  #内部扩张2倍
        self.mlp_ratio = mlp_ratio
        self.att_flag = att
        self.Tsteps = timestep
        self.lif1 = mem_update()
        self.lif2 = mem_update()

        self.conv1 = RepConv(input_dim, int(input_dim * mlp_ratio)) #137以外的模型，在第一个block不做分组

        self.bn1 = nn.BatchNorm2d(int(input_dim * mlp_ratio))  # 这里可以进行改进

        self.conv2 = RepConv(int(input_dim * mlp_ratio), input_dim)
        self.bn2 = nn.BatchNorm2d(input_dim)  # 这里可以进行改进
        if(self.att_flag == 1):
            self.att = TCSA(self.Tsteps,input_dim)
        elif(self.att_flag == 2):
            self.att = Spike_TCSA(self.Tsteps,input_dim)

    # @get_local('x_feat')
    def forward(self, x):
        T, B, C, H, W = x.shape
        # print(f"MS_ConvBlockinput = {x.shape}")
        x = self.Conv(x) + x  #sepconv  pw+dw+pw
        x_feat = x
        x = self.bn1(self.conv1(self.lif1(x).flatten(0, 1))).reshape(T, B, int(self.mlp_ratio * C), H, W)
        x = self.bn2(self.conv2(self.lif2(x).flatten(0, 1))).reshape(T, B, C, H, W)
        x = x_feat + x
        if(self.att_flag):
            x,ta = self.att(x)
        # print(f"MS_ConvBlockout = {x.shape}")
        return x

class Att_MS_ConvBlock(nn.Module):
    def __init__(self, input_dim, mlp_ratio=4.,sep_kernel_size = 7 ,att = 0,timestep = 4,full=False):  # in_channels(out_channels), 内部扩张比例
        super().__init__()

        self.full =full
        self.Conv = SepConv(dim=input_dim,kernel_size= sep_kernel_size)  #内部扩张2倍
        self.mlp_ratio = mlp_ratio
        self.att_flag = att
        self.Tsteps = timestep
        self.lif1 = mem_update()
        self.lif2 = mem_update()

        self.conv1 = RepConv(input_dim, int(input_dim * mlp_ratio)) #137以外的模型，在第一个block不做分组

        self.bn1 = nn.BatchNorm2d(int(input_dim * mlp_ratio))  # 这里可以进行改进

        self.conv2 = RepConv(int(input_dim * mlp_ratio), input_dim)
        self.bn2 = nn.BatchNorm2d(input_dim)  # 这里可以进行改进
        if(self.att_flag == 1):
            self.att = TCSA(self.Tsteps,input_dim)
        elif(self.att_flag == 2):
            self.att = Spike_TCSA(self.Tsteps,input_dim)


    # @get_local('x_feat')
    def forward(self, x,t_all):
        T, B, C, H, W = x.shape
        # print(f"MS_ConvBlockinput = {x.shape}")
        x = self.Conv(x) + x  #sepconv  pw+dw+pw

        x_feat = x
        if(self.att_flag):
            x = self.bn1(self.conv1(self.lif1(x).flatten(0, 1))).reshape(T, B, int(self.mlp_ratio * C), H, W)
            x = self.bn2(self.conv2(self.lif2(x).flatten(0, 1))).reshape(T, B, C, H, W)
            x,ta = self.att(x)
            # 对第一维度取平均值
            ta_mean = ta.mean(dim=0)  # 结果形状为 torch.Size([4, 1, 1, 1]
            for i in range(4):
                t_all[i] += (ta_mean[i] / 3).item()             # 对第一维度取平均值
            
        else:
            x = self.bn1(self.conv1(self.lif1(x).flatten(0, 1))).reshape(T, B, int(self.mlp_ratio * C), H, W)
                #repconv，对应conv_mixer，包含1*1,3*3,1*1三个卷积，等价于一个3*3卷积
            x = self.bn2(self.conv2(self.lif2(x).flatten(0, 1))).reshape(T, B, C, H, W)

        x = x_feat + x
        # print(f"MS_ConvBlockout = {x.shape}")
        return x


## 时间通道注意力
class TemporalChannelAttention(nn.Module):
    def __init__(self, T, in_channels, reduction=4):
        super(TemporalChannelAttention, self).__init__()
        # r 用来设置扩展或压缩倍数
        # Linear layers for time and channel attention
        self.fc_time_expand = nn.Linear(T, T * reduction, bias=False)
        self.fc_time_compress = nn.Linear(T * reduction, T, bias=False)
        self.fc_channel_expand = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc_channel_compress = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 输入 x: [T, B, C, H, W]
        T, B, C, H, W = x.shape
        x_feat = x
        # 1. 对 H, W 维度进行全局平均池化得到 [T, B, C]
        x_pool = x.mean(dim=(3, 4))  # [T, B, C]

        # 2. 计算时间注意力：先在 C 维度上最大池化，再进行扩展和压缩
        time_attention = x_pool.max(dim=2)[0].transpose(0, 1)  # [B, T]
        time_attention = self.fc_time_expand(time_attention)  # 扩展维度到 [B, T*reduction]
        time_attention = torch.relu(time_attention)           # 激活函数
        time_attention = self.fc_time_compress(time_attention)  # 压缩维度回 [B, T]
        time_attention = self.sigmoid(time_attention)         # 使用 sigmoid 归一化
        time_attention = time_attention.transpose(0, 1).unsqueeze(-1)       # 转换为 [T, B, 1]，以便广播

        # 3. 计算通道注意力：在 T 维度上最大池化，再进行扩展和压缩
        channel_attention = x_pool.max(dim=0)[0]  # [B, C]
        channel_attention = self.fc_channel_expand(channel_attention)  # 扩展维度到 [B, C//reduction]
        channel_attention = torch.relu(channel_attention)              # 激活函数
        channel_attention = self.fc_channel_compress(channel_attention)  # 压缩维度回 [B, C]
        channel_attention = self.sigmoid(channel_attention).unsqueeze(-1)             # 使用 sigmoid 归一化

        #print(f"time_attention = {time_attention.unsqueeze(-1).unsqueeze(-1) .shape} channel_attention = {channel_attention.shape}")
        # 4. 将注意力应用到原始特征上
        x = x * time_attention.unsqueeze(-1).unsqueeze(-1)   # [T, B, C, H, W] * [T, B, 1, 1, 1]
        x = x * channel_attention.unsqueeze(0).unsqueeze(-1)  # [T, B, C, H, W] * [1, B, C, 1, 1]

        return x_feat + x

## 时间空间注意力
class TemporalSpatialAttention(nn.Module):
    def __init__(self,T,reduction_ratio = 4,
                 dilation_conv_num=2,
                 dilation_val=4):
        super(TemporalSpatialAttention, self).__init__()
        # 时间注意力
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.gate_t = nn.Sequential(
            nn.Linear(T, T * reduction_ratio),
            nn.BatchNorm1d(T * reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(T * reduction_ratio,T),
        )
        reduced_c = T * reduction_ratio
        self.gate_s = nn.Sequential()

        self.gate_s.add_module(
            'gate_s_conv_reduce0',
            nn.Conv2d(T, reduced_c, kernel_size=1))
        self.gate_s.add_module('gate_s_bn_reduce0',
                               nn.BatchNorm2d(reduced_c))
        self.gate_s.add_module('gate_s_relu_reduce0', nn.ReLU())

        # 进行多个空洞卷积，丰富感受野
        for i in range(dilation_conv_num):
            self.gate_s.add_module(
                'gate_s_conv_di_%d' % i,
                nn.Conv2d(reduced_c, reduced_c,
                          kernel_size=3,
                          padding=dilation_val,
                          dilation=dilation_val))
            self.gate_s.add_module(
                'gate_s_bn_di_%d' % i,
                nn.BatchNorm2d(reduced_c))
            self.gate_s.add_module('gate_s_relu_di_%d' % i, nn.ReLU())

        self.gate_s.add_module(
            'gate_s_conv_final',
            nn.Conv2d(reduced_c, 1, kernel_size=1))

    def forward(self, x):
        # x: [T, B, C, H, W]
        T, B, C, H, W = x.shape
        x_feat = x
        x_avg = torch.mean(x, dim=2).transpose(0, 1)  # Channel-averaged [B, T, H, W]
        time_attention = self.avgpool(x_avg).squeeze(-1).squeeze(-1)
        time_attention = self.gate_t(time_attention)
        # print(f"time_attention = {time_attention.shape}")

        spatial_attention = self.gate_s(x_avg)
        # print(f"spatial_attention = {spatial_attention.shape}") # [B,T,H,W]

        x = x * time_attention.transpose(0, 1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)   # [T, B, C, H, W] * [T, B]
        x = x * spatial_attention.transpose(0, 1).unsqueeze(2)  # [T, B, C, H, W] * [T, B, 1, H, W]
        
        return x + x_feat  # Broadcast and apply attention

## T-CBAM
class TemporalSpatialChannelAttention(nn.Module):
    def __init__(self,T, in_channels, reduction = 4, apply_temporal_channel=True, apply_temporal_spatial=True):
        super(TemporalSpatialChannelAttention, self).__init__()
        self.apply_temporal_channel = apply_temporal_channel
        self.apply_temporal_spatial = apply_temporal_spatial
        if apply_temporal_channel:
            self.temporal_channel_attention = TemporalChannelAttention(T, in_channels, reduction=4)
        if apply_temporal_spatial:
            self.temporal_spatial_attention = TemporalSpatialAttention(T,reduction_ratio=4)

    def forward(self, x):
        if self.apply_temporal_channel:
            x = x + self.temporal_channel_attention(x)
        if self.apply_temporal_spatial:
            x = x + self.temporal_spatial_attention(x)
        return x


class MS_AllConvBlock(nn.Module):  # standard conv
    def __init__(self, input_dim, mlp_ratio=4.,sep_kernel_size = 7,timestep = 4,group=False):  # in_channels(out_channels), 内部扩张比例
        super().__init__()

        self.Conv = SepConv(dim=input_dim,kernel_size= sep_kernel_size)
        self.Tsteps = timestep
        self.mlp_ratio = mlp_ratio
        self.conv1 = MS_StandardConv(input_dim, int(input_dim * mlp_ratio),3)
        self.conv2 = MS_StandardConv(int(input_dim * mlp_ratio), input_dim,3)

    # @get_local('x_feat')
    def forward(self, x):
        T, B, C, H, W = x.shape
        # print(f"MS_AllConvBlock_input = {x.shape}")
        x = self.Conv(x) + x  #sepconv  pw+dw+pw

        x_feat = x

        x = self.conv1(x)
        x = self.conv2(x)

        x = x_feat + x
        # print(f"MS_AllConvBlock_output = {x.shape}")
        return x

class att_MS_AllConvBlock(nn.Module):  # standard conv
    def __init__(self, input_dim, mlp_ratio=4.,sep_kernel_size = 7,att = 0,timestep = 4,group=False):  # in_channels(out_channels), 内部扩张比例
        super().__init__()

        self.Conv = SepConv(dim=input_dim,kernel_size= sep_kernel_size)
        self.att_flag = att
        self.Tsteps = timestep
        self.mlp_ratio = mlp_ratio
        self.conv1 = MS_StandardConv(input_dim, int(input_dim * mlp_ratio),3)
        self.conv2 = MS_StandardConv(int(input_dim * mlp_ratio), input_dim,3)
        if(self.att_flag == 1):
            self.att = TCSA(self.Tsteps,input_dim)
        elif(self.att_flag == 2):
            self.att = Spike_TCSA(self.Tsteps,input_dim)

    # @get_local('x_feat')
    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.Conv(x) + x  #sepconv  pw+dw+pw

        x_feat = x

        x = self.conv1(x)
        x = self.conv2(x)

        x = x_feat + x
        if self.att_flag:
            x,ta = self.att(x)
        # print(f"MS_AllConvBlock_output = {x.shape}")
        return x

class Att_MS_AllConvBlock(nn.Module):  # standard conv
    def __init__(self, input_dim, mlp_ratio=4.,sep_kernel_size = 7,att = 0,timestep = 4,group=False):  # in_channels(out_channels), 内部扩张比例
        super().__init__()

        self.Conv = SepConv(dim=input_dim,kernel_size= sep_kernel_size)
        self.att_flag = att
        self.Tsteps = timestep
        self.mlp_ratio = mlp_ratio
        self.conv1 = MS_StandardConv(input_dim, int(input_dim * mlp_ratio),3)
        self.conv2 = MS_StandardConv(int(input_dim * mlp_ratio), input_dim,3)
        if(self.att_flag == 1):
            self.att = TCSA(self.Tsteps,input_dim)
        elif(self.att_flag == 2):
            self.att = Spike_TCSA(self.Tsteps,input_dim)

    # @get_local('x_feat')
    def forward(self, x,t_all):
        T, B, C, H, W = x.shape
        x = self.Conv(x) + x  #sepconv  pw+dw+pw

        x_feat = x

        if self.att_flag:
            x = self.conv1(x)
            x = self.conv2(x)
            x,ta = self.att(x)
            # 对第一维度取平均值
            ta_mean = ta.mean(dim=0)  # 结果形状为 torch.Size([4, 1, 1, 1]
            for i in range(4):
                t_all[i] += (ta_mean[i] / 3).item() 
            
        else:
            x = self.conv1(x)
            x = self.conv2(x)

        x = x_feat + x
        # print(f"MS_AllConvBlock_output = {x.shape}")
        return x

class MS_StandardConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1):  # in_channels(out_channels), 内部扩张比例
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.s = s
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.lif = mem_update()

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.bn(self.conv(self.lif(x).flatten(0, 1))).reshape(T, B, self.c2, int(H / self.s), int(W / self.s))
        return x

class MS_DownSampling(nn.Module):
    def __init__(self, in_channels=2, embed_dims=256, kernel_size=3, stride=2, padding=1, first_layer=True):
        super().__init__()

        self.encode_conv = nn.Conv2d(in_channels, embed_dims, kernel_size=kernel_size, stride=stride, padding=padding)

        self.encode_bn = nn.BatchNorm2d(embed_dims)
        if not first_layer:
            self.encode_lif = mem_update()
        # self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        T, B, _, _, _ = x.shape

        # print(f"x = {x.shape}")
        if hasattr(self, "encode_lif"): #如果不是第一层
            # x_pool = self.pool(x)
            x = self.encode_lif(x)

        # print(f"x = {x.device}")
        x = self.encode_conv(x.flatten(0, 1))
        _, C, H, W = x.shape
        x = self.encode_bn(x).reshape(T, B, -1, H, W).contiguous()

        # if not hasattr(self, "encode_lif"):
        #     print("xixix")
        #     for t in range(T):
        #         temp = x[t]
        #         average_image = temp[0].mean(dim=0).detach().cpu().numpy()
        #         # 使用 matplotlib 显示平均后的图像
        #         plt.imshow(average_image, cmap='gray')
        #         plt.axis('off')  # 不显示坐标轴
        #         plt.savefig(f"./result/average_feature_{t}.png", dpi=300, bbox_inches='tight')  # 保存为图片
        #         plt.close()  # 关闭图像以避免显示
        return x

class MS_GetT(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, T=4):
        super().__init__()
        self.T = T
        self.in_channels = in_channels

    def forward(self, x):
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        return x
    

class generate_T(nn.Module):
    def __init__(self, in_channels=2, embed_dims=256, kernel_size=3, stride=2, padding=1):
        super().__init__()

        self.encode_conv = nn.Conv2d(in_channels, embed_dims, kernel_size=kernel_size, stride=stride, padding=padding)
        self.encode_bn = nn.BatchNorm2d(embed_dims)
        self.act = nn.ReLU()  # 修正为赋值

    def forward(self, x):
        B, _, _, _ = x.shape

        x_feat = x
        x = self.act(self.encode_bn(self.encode_conv(x)))

        return x_feat + x
    

## 捷径连接，注意力分数
class Co_Diff_GetT(nn.Module):
    def __init__(self, in_channels=2, embed_dims=256, kernel_size=3, stride=2, padding=1, T=4,t_kernel_size = 3,t_stride = 1,t_pading = 1):
        super().__init__()
        self.T = T
        self.embed_dims = embed_dims
        self.in_channels = in_channels

        # 实例化 T-1 个 generate_T 对象
        self.time_modules = nn.ModuleList([
            generate_T(embed_dims, embed_dims, t_kernel_size, t_stride, t_pading)
            for _ in range(T)
        ])

        # 最后一层卷积
        self.encode_conv = nn.Conv2d(in_channels, embed_dims, kernel_size=kernel_size, stride=stride, padding=padding)
        self.encode_bn = nn.BatchNorm2d(embed_dims)
        self.act = nn.ReLU()  # 修正为赋值

        # 可训练参数，用于调整 base_feature 和 current_feature 的权重
        self.alpha_base = nn.Parameter(torch.tensor(1.0))   # 初始化为 1.0
        self.alpha_current = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, t_all):
        B, C, H, W = x.shape  # 输入形状没有时间维度

        # 应用最后一层卷积作为初始特征提取
        base_feature = self.act(self.encode_bn(self.encode_conv(x)))  # (B, embed_dims, H', W')
        # print(f"base_feature = {base_feature.shape}")

        # # 初始化输出列表
        # outputs = [base_feature.unsqueeze(0)]  # 添加第一个时间步的特征
        outputs = []
        # 逐步通过时间模块生成剩余时间步特征 记得检查目前是否采用了快捷连接
        current_feature = base_feature
        for t in range(self.T):
            current_feature = self.time_modules[t](current_feature)  # 通过对应时间步模块
            t_all_t = t_all[t].to(current_feature.device)
            current_feature = self.alpha_base * base_feature * (1 - t_all_t) + self.alpha_current * current_feature * t_all_t
            outputs.append(current_feature.unsqueeze(0))  # 添加时间维度
            # average_image = current_feature[0].mean(dim=0).detach().cpu().numpy()

            # # 使用 matplotlib 显示平均后的图像
            # plt.imshow(average_image, cmap='gray')
            # plt.axis('off')  # 不显示坐标轴
            # plt.savefig(f"./result/average_feature_{t}.png", dpi=300, bbox_inches='tight')  # 保存为图片
            # plt.close()  # 关闭图像以避免显示

        # 将所有时间步的输出组合
        return torch.cat(outputs, dim=0)  # 输出形状为 (T, B, C, H, W)

# 捷径连接
class Diff_GetT(nn.Module):
    def __init__(self, in_channels=2, embed_dims=256, kernel_size=3, stride=2, padding=1, T=4,t_kernel_size = 3,t_stride = 1,t_pading = 1):
        super().__init__()
        self.T = T
        self.embed_dims = embed_dims
        self.in_channels = in_channels

        self.time_modules = nn.ModuleList([
            generate_T(embed_dims, embed_dims, t_kernel_size, t_stride, t_pading)
            for _ in range(T)
        ])

        # 最后一层卷积
        self.encode_conv = nn.Conv2d(in_channels, embed_dims, kernel_size=kernel_size, stride=stride, padding=padding)
        self.encode_bn = nn.BatchNorm2d(embed_dims)
        self.act = nn.ReLU()  # 修正为赋值

    def forward(self, x):
        B, C, H, W = x.shape  # 输入形状没有时间维度

        # 应用最后一层卷积作为初始特征提取
        base_feature = self.act(self.encode_bn(self.encode_conv(x)))  # (B, embed_dims, H', W')
        # print(f"base_feature = {base_feature.shape}")

        # # 初始化输出列表
        # outputs = [base_feature.unsqueeze(0)]  # 添加第一个时间步的特征
        outputs = []
        # 逐步通过时间模块生成剩余时间步特征 记得检查目前是否采用了快捷连接
        current_feature = base_feature

        for t in range(self.T):
            current_feature = self.time_modules[t](current_feature)  # 通过对应时间步模块
            current_feature = base_feature + current_feature
            outputs.append(current_feature.unsqueeze(0))  # 添加时间维度

            # average_image = current_feature[0].mean(dim=0).detach().cpu().numpy()
            # # 使用 matplotlib 显示平均后的图像
            # plt.imshow(average_image, cmap='gray')
            # plt.axis('off')  # 不显示坐标轴
            # plt.savefig(f"./result/average_feature_{t}.png", dpi=300, bbox_inches='tight')  # 保存为图片
            # plt.close()  # 关闭图像以避免显示

        # 将所有时间步的输出组合
        return torch.cat(outputs, dim=0)  # 输出形状为 (T, B, C, H, W)

class MS_CancelT(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, T=2):
        super().__init__()
        self.T = T

    def forward(self, x):
        x = x.mean(0)
        return x

class SpikeConv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.lif = mem_update()
        self.bn = nn.BatchNorm2d(c2)
        self.s = s
        # self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        T, B, C, H, W = x.shape
        H_new = int(H / self.s)
        W_new = int(W / self.s)
        x = self.lif(x)
        x = self.bn(self.conv(x.flatten(0, 1))).reshape(T, B, -1, H_new, W_new)
        return x

class SpikeConvWithoutBN(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=True)
        self.lif = mem_update()
        self.s = s
        # self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        T, B, C, H, W = x.shape
        H_new = int(H / self.s)
        W_new = int(W / self.s)
        x = self.lif(x)
        x = self.conv(x.flatten(0, 1)).reshape(T, B, -1, H_new, W_new)
        return x

class SpikeSPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = SpikeConv(c1, c_, 1, 1)
        self.cv2 = SpikeConv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            T, B, C, H, W = x.shape
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x.flatten(0, 1)).reshape(T, B, -1, H, W)
            y2 = self.m(y1.flatten(0, 1)).reshape(T, B, -1, H, W)
            y3 = self.m(y2.flatten(0, 1)).reshape(T, B, -1, H, W)
            return self.cv2(torch.cat((x, y1, y2, y3), 2))



class MS_Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):  # 这里输入x是一个list
        for i in range(len(x)):
            if x[i].dim() == 5:
                x[i] = x[i].mean(0)
        return torch.cat(x, self.d)

if __name__ == "__main__":
    T, B, C, H, W = 8, 4, 16, 32, 32
    x = torch.randn(T, B, C, H, W)
    attention_layer = TemporalSpatialChannelAttention(T=8,in_channels=C)
    output = attention_layer(x)
    print("Output shape:", output.shape)  # 应该是 (T, B, C, H, W)