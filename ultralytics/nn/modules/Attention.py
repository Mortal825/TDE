import torch
from torch import nn
from einops import rearrange, repeat
import torch.nn.functional as F
import numpy as np

class TimeAttention(nn.Module):
    def __init__(self, in_planes, ratio=16, fc=False):
        super(TimeAttention, self).__init__()
        if fc ==True:
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
            self.max_pool = nn.AdaptiveMaxPool1d(1)

            self.sharedMLP = nn.Sequential(
                nn.Conv1d(in_planes, in_planes * ratio, 1, bias=False),
                nn.ReLU(),
                nn.Conv1d(in_planes * ratio, in_planes, 1, bias=False),
            )
        else:
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
            self.max_pool = nn.AdaptiveMaxPool3d(1)

            self.sharedMLP = nn.Sequential(
                nn.Conv3d(in_planes, in_planes * ratio, 1, bias=False),
                nn.ReLU(),
                nn.Conv3d(in_planes * ratio, in_planes, 1, bias=False),
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class Spike_TimeAttention(nn.Module):
    def __init__(self, in_planes, ratio=16, fc=False):
        super(Spike_TimeAttention, self).__init__()
        from ultralytics.nn.modules.yolo_spikformer import getmem_update,nomem_update
        if fc ==True:
            self.max_pool = nn.AdaptiveMaxPool1d(1)
            self.lif1 = nomem_update()
            self.lif2 = getmem_update()
            self.conv1 = nn.Conv1d(in_planes, in_planes, 1, bias=False)


        else:
            self.max_pool = nn.AdaptiveMaxPool3d(1)
            self.lif1 = nomem_update()
            self.lif2 = getmem_update()

            self.conv1 = nn.Conv3d(in_planes, in_planes, 1, bias=False)

    def forward(self, x):
        x = self.max_pool(x)
        x = self.lif1(x)
        x = self.conv1(x)
        spike_out,float_out = self.lif2(x)
        return spike_out,float_out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
    

        self.sharedMLP = nn.Sequential(
            nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = rearrange(x, "b t c h w -> b c t h w")
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        out = self.sigmoid(avgout + maxout)
        out = rearrange(out, "b c t h w -> b t c h w")
        return out
    

class Spike_ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(Spike_ChannelAttention, self).__init__()
        from ultralytics.nn.modules.yolo_spikformer import getmem_update,nomem_update
        self.max_pool = nn.AdaptiveMaxPool3d(1)
    
        self.lif1 = nomem_update()
        self.lif2 = getmem_update()
        self.conv1 = nn.Conv3d(in_planes, in_planes, 1, bias=False)

    def forward(self, x):
        x = rearrange(x, "b t c h w -> b c t h w")
        x = self.max_pool(x)
        x = self.lif1(x)
        x = self.conv1(x)
        spike_out,float_out = self.lif2(x)
        spike_out = rearrange(spike_out, "b t c h w -> b c t h w")
        float_out = rearrange(float_out, "b t c h w -> b c t h w")
        return spike_out,float_out



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        c = x.shape[2]
        x = rearrange(x, "b t c h w -> b (t c) h w")
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        x = x.unsqueeze(1)

        return self.sigmoid(x)

class Spike_SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(Spike_SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        from ultralytics.nn.modules.yolo_spikformer import getmem_update,nomem_update
        self.lif1 = nomem_update()
        self.lif2 = getmem_update()

        self.conv = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        c = x.shape[2]
        x = rearrange(x, "b t c h w -> b (t c) h w")
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        # maxout = torch.mean(x, dim=1, keepdim=True)
        # print("maxout",maxout)
        x = self.lif1(maxout)
        x = self.conv(x)

        spike_out,float_out = self.lif2(x)
        spike_out = spike_out.unsqueeze(1)
        float_out = float_out.unsqueeze(1)
        return spike_out,float_out



import matplotlib.pyplot as plt
import torch


def save_avg_out_heatmap(out_tensor, save_path="out_heatmap.png", gamma=2.0):
    """
    保存反转并增强对比度的热力图：低值更亮、更突出，非重点区域更暗。
    :param gamma: γ > 1 时，低值更加突出
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    # 平均前3维 -> shape: (H, W)
    avg_map = out_tensor.mean(dim=(0, 1, 2)).detach().cpu().numpy()

    # 归一化并反转
    norm_map = (avg_map - np.min(avg_map)) / (np.max(avg_map) - np.min(avg_map) + 1e-8)
    inv_map = 1.0 - norm_map  # 低值更亮

    # 加 gamma 映射增强对比度（非线性）
    contrast_map = inv_map ** gamma

    # 绘制热力图
    plt.figure(figsize=(8, 6))
    plt.imshow(contrast_map, cmap="magma", interpolation='nearest', vmin=0, vmax=1)
    cbar = plt.colorbar(label="Activation", fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    plt.title(f"Activation Map (gamma={gamma})", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.close()

class TCSA(nn.Module):
    def __init__(self, timeWindows, channels, stride=1, c_ratio=16, t_ratio=16):
        super(TCSA, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(channels, c_ratio)
        self.ta = TimeAttention(timeWindows, t_ratio)
        self.sa = SpatialAttention()

        self.stride = stride

    def forward(self, x):
        # print("hahah")
        x = rearrange(x, "t b c h w -> b t c h w")
        ta = self.ta(x)
        out = ta * x
        out = self.ca(out) * out  # 广播机制
        out = self.sa(out) * out  # 广播机制

        out = self.relu(out)
        out = rearrange(out, "b t c h w -> t b c h w")
        # save_avg_out_heatmap(out,save_path="tcs_out_heatmap.png")
        return out,ta
    
class Spike_TCSA(nn.Module):
    def __init__(self, timeWindows, channels, stride=1, c_ratio=16, t_ratio=1):
        super(Spike_TCSA, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.ca = Spike_ChannelAttention(channels, c_ratio)
        self.ta = Spike_TimeAttention(timeWindows, t_ratio)
        self.sa = Spike_SpatialAttention()

        self.stride = stride

    def forward(self, x):
        x = rearrange(x, "t b c h w -> b t c h w")
        ta_spike,ta_float = self.ta(x)
        ca_spike,ca_float = self.ca(x)
        sa_spike,sa_float = self.sa(x)

        att1  = ta_spike * sa_float * ca_spike
        att2  = ta_spike * ca_float * sa_spike
        att3  = sa_spike * ta_float * ca_spike
        out = x + (att1 + att2 + att3) / 3
        out = rearrange(out, "f b c h w -> b f c h w")
        # print("out",out.shape)
        # save_avg_out_heatmap(out,save_path="spike_tcs_out_heatmap.png")
        return out,ta_float

