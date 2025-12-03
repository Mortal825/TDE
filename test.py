import os
from ultralytics import YOLO
import torch.nn as nn
import torch
import pandas as pd
# from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode
from ultralytics.nn.modules.yolo_spikformer import MultiSpike4,MultiSpike2,MultiSpike1,mem_update,Time_Decoder,getmem_update,nomem_update
from ultralytics.nn.modules.Attention import TimeAttention
from collections import Counter
import matplotlib.pyplot as plt
from torchsummary import summary  #torchsummary.torchsummary
import torchsummary
print(torchsummary.__file__)
import numpy as np

i = 0
Att = {}
small_att = []
mid_att = []
large_att = []
fr_dict = {}

# 假设时间步为 4，可根据实际情况调整
T = 4

# 初始化 combination_dict
combination_dict_last = {bin(i)[2:].zfill(T): 0 for i in range(2**T)}
combination_dict_first = {bin(i)[2:].zfill(T): 0 for i in range(2**T)}

iter = 2518 #the iters of an epoch

## VOC2007
## baseline
# model = YOLO("./runs/detect/VOC07/Baseline/weights/best.pt")
# ## TCSA
# model = YOLO("./runs/detect/VOC07/TCSA/weights/best.pt")
# ## SDA
# model = YOLO("./runs/detect/VOC07/SDA/weights/best.pt")
# ## SE
# model = YOLO("./runs/detect/VOC07/SE/weights/best.pt")
# ## TDE(TCSA)
# model = YOLO("./runs/detect/VOC07/TDE(TCSA)/weights/best.pt")
## TDE(SDA)
# model = YOLO("./runs/detect/VOC07/TDE(SDA)/weights/best.pt")

## EVDET200K
## baseline 
# model = YOLO('./runs/detect/EVDet200K/baseline/weights/best.pt')
## TDE(TCSA)
# model = YOLO('./runs/detect/EVDet200K/TDE(TCSA)/weights/best.pt')
## TDE(SDA)
model = YOLO('./runs/detect/EVDet200K/TDE(SDA)/weights/best.pt')

## VOC
## baseline 
# model = YOLO('./runs/detect/VOC/baseline/weights/best.pt')
## TDE(SDA)
# model = YOLO('./runs/detect/VOC/TDE(SDA)/weights/best.pt')
## TDE(TCSA)
# model = YOLO('./runs/detect/VOC/TDE(TCSA)/weights/best.pt')

print(f"model = {model}")
def forward_hook_fn(module, input, output):  # 计算每一层的发放率
    if(module.name == "model.model.18.att.ta"):
        # print(f"input = {input[0].shape}")  
        T,_,_,_ = output[0].shape
        flattened_tensor = output[0].detach().reshape(T, -1)
        small_att.append(list(flattened_tensor.cpu().numpy()))
        if "small" not in Att.keys():
            Att["small"] = flattened_tensor / iter
        else:
            Att["small"] += flattened_tensor / iter
    if(module.name == "model.model.21.att.ta"):
        T,_,_,_ = output[0].shape
        flattened_tensor = output[0].detach().reshape(T, -1)
        mid_att.append(list(flattened_tensor.cpu().numpy())) 
        if "mid" not in Att.keys():
            Att["mid"] = flattened_tensor / iter
        else:
            Att["mid"] += flattened_tensor / iter
    if(module.name == "model.model.24.att.ta"):
        T,_,_,_ = output[0].shape
        flattened_tensor = output[0].detach().reshape(T, -1) 
        large_att.append(list(flattened_tensor.cpu().numpy()))
        if "large" not in Att.keys():
            Att["large"] = flattened_tensor / iter
        else:
            Att["large"] += flattened_tensor / iter



def forward_hook_fn_fr(module, input, output):  
    if(module.name == "model.model.25.cv2.0.0.lif"):
        global combination_dict_last
        T, N, C, H, W = output.shape
        n_sum = N * C * H * W  # 总的样本数
        # 确保设备一致
        device = output.device

        # Step 1: 展平维度 [T, N, C, H, W] -> [T, X]
        flattened_tensor = output.detach().reshape(T, -1)  # Shape: [T, X]

        # Step 2: 转置为 [X, T]，每行是一个组合
        combinations = flattened_tensor.T  # Shape: [X, T]

        # Step 3: 将每个组合看作二进制数，并转换为十进制编号
        weights = (2 ** torch.arange(T - 1, -1, -1, dtype=torch.float, device=device))
        indices = combinations @ weights
        indices = indices.to(torch.long)  # 转换为整型以供 bincount 使用

        # Step 4: 统计每种组合的出现次数
        counts = torch.bincount(indices, minlength=2**T)

        for i, count in enumerate(counts.tolist()):
            frequency = count / n_sum  # 归一化频率
            binary_rep = bin(i)[2:].zfill(T)  # 转换为二进制字符串表示
            combination_dict_last[binary_rep] += frequency/ iter  # 存储到字典中

    if(module.name == "model.model.1.Conv.lif1"):
        global combination_dict_first
        T, N, C, H, W = output.shape
        n_sum = N * C * H * W  # 总的样本数
        # 确保设备一致
        device = output.device

        # Step 1: 展平维度 [T, N, C, H, W] -> [T, X]
        flattened_tensor = output.detach().reshape(T, -1)  # Shape: [T, X]

        # Step 2: 转置为 [X, T]，每行是一个组合
        combinations = flattened_tensor.T  # Shape: [X, T]

        # Step 3: 将每个组合看作二进制数，并转换为十进制编号
        weights = (2 ** torch.arange(T - 1, -1, -1, dtype=torch.float, device=device))
        indices = combinations @ weights
        indices = indices.to(torch.long)  # 转换为整型以供 bincount 使用

        # Step 4: 统计每种组合的出现次数
        counts = torch.bincount(indices, minlength=2**T)

        for i, count in enumerate(counts.tolist()):
            frequency = count / n_sum  # 归一化频率
            binary_rep = bin(i)[2:].zfill(T)  # 转换为二进制字符串表示
            combination_dict_first[binary_rep] += frequency/ iter  # 存储到字典中

    global fr_dict  # 确保 fr_dict 是全局变量

    if module.name not in fr_dict.keys():
        if (type(output) == tuple):
            output = output[0]
        print("module.name = ", module.name)
        fr_dict[module.name] = {
            "firing_rate": 0.0,  # 计算发放率
        }
    else:
        if (type(output) == tuple):
            output = output[0]
        if(output.dim() == 5):
            fr_dict[module.name]["firing_rate"] += output.detach().mean().item() / iter
            fr_dict[module.name]["tensor_shape"] = [output.shape[0],output.shape[1], output.shape[2],output.shape[3],output.shape[4]]
        if(output.dim() == 4):
            fr_dict[module.name]["firing_rate"] += output.detach().mean().item() / iter
            fr_dict[module.name]["tensor_shape"] = [output.shape[0],output.shape[1], output.shape[2],output.shape[3]]


for n, m in model.named_modules():
    # print("m = ",m)
    if isinstance(m, Time_Decoder):
        # print(n)  
        for param_name, param_value in m.named_parameters():
            print(f"Parameter name: {param_name}, Parameter value: {param_value}")  ##打印编码层参数 1.53 2.01 1.61 -3.28  bias 1.109

    if isinstance(m, mem_update) or isinstance(m, torch.nn.Identity) or isinstance(m, getmem_update) or isinstance(m, nomem_update):
        print(n)
        m.name = n
        m.register_forward_hook(forward_hook_fn_fr) 

    if isinstance(m, TimeAttention):
        # print(n)
        m.name = n
        m.register_forward_hook(forward_hook_fn)


model.val(data="./config/EvDET200K.yaml",device=[2])  
# model.val(data="./config/VOC.yaml",device=[2])  
# model.val(data="./config/VOC07.yaml",device=[2]) 

print("fire:",fr_dict) #the firing rate of each layer
# 将 fr_dict 转换为 DataFrame
data = []
for layer_name, info in fr_dict.items():
    data.append([
        layer_name,
        info["firing_rate"],
        str(info["tensor_shape"])  # 将 tensor shape 转换为字符串存储
    ])

# 创建 Pandas DataFrame
df = pd.DataFrame(data, columns=["Layer Name", "Firing Rate", "Tensor Shape"])
print("df:",len(df))
# 保存为 Excel 文件
excel_path = "firing_rates.xlsx"
df.to_excel(excel_path, index=False)

print(f"Excel 文件已保存到: {excel_path}")
print(f"att = {Att}")

# === 首层直方图 ===
combinations = list(combination_dict_first.keys())
frequencies = list(combination_dict_first.values())

plt.figure(figsize=(10, 6))
plt.bar(combinations, frequencies, width=0.6, align='center', color='#4A90E2')
plt.xlabel("Binary Combinations", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.title("Frequency of Binary Combinations (First Layer)", fontsize=14)
plt.xticks(rotation=45, fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# === 添加数值标注 ===
for i, (comb, freq) in enumerate(zip(combinations, frequencies)):
    plt.text(i, freq + max(frequencies) * 0.01, f"{freq:.2f}", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(f"first_layer_binary_combinations.png", dpi=300, bbox_inches='tight')
plt.close()

# 保存首层直方图数据
df_first = pd.DataFrame({
    "Binary Combination": list(combination_dict_first.keys()),
    "Frequency": list(combination_dict_first.values())
})
df_first.to_csv(f"first_layer_binary_combinations.csv", index=False)



# === 最后一层直方图 ===
combinations = list(combination_dict_last.keys())
frequencies = list(combination_dict_last.values())

plt.figure(figsize=(10, 6))
plt.bar(combinations, frequencies, width=0.6, align='center', color='#4A90E2')
plt.xlabel("Binary Combinations", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.title("Frequency of Binary Combinations (First Layer)", fontsize=14)
plt.xticks(rotation=45, fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# === 添加数值标注 ===
for i, (comb, freq) in enumerate(zip(combinations, frequencies)):
    plt.text(i, freq + max(frequencies) * 0.01, f"{freq:.2f}", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(f"last_layer_binary_combinations.png", dpi=300, bbox_inches='tight')
plt.close()

# 保存最后一层直方图数据
df_last = pd.DataFrame({
    "Binary Combination": list(combination_dict_last.keys()),
    "Frequency": list(combination_dict_last.values())
})
df_last.to_csv(f"last_layer_binary_combinations.csv", index=False)

## 绘制small_att,mid_att,large_att的折线图
small_att = np.array(small_att)
mid_att = np.array(mid_att)
large_att = np.array(large_att)
# 创建一个包含 4 个子图的图形，4行1列
# 创建一个包含 4 个子图的图形，4行1列
fig, axes = plt.subplots(4, 1, figsize=(10, 12))

# 如果只有一列或一行，将 axes 变为一维数组
if len(axes.shape) == 1:
    axes = axes.flatten()

# 绘制每个时间步（子图），每个子图上绘制3条折线：small_att, mid_att, large_att
for i in range(4):
    axes[i].plot(small_att[:, i], label='Small Att', marker='o')
    axes[i].plot(mid_att[:, i], label='Mid Att', marker='s')
    axes[i].plot(large_att[:, i], label='Large Att', marker='^')
    
    axes[i].set_title(f'Time Step {i+1}')
    axes[i].set_xlabel('Batch')
    axes[i].set_ylabel('Attention Value')
    axes[i].legend()

# 调整布局，避免子图重叠
fig.tight_layout()

# 保存图形为文件
plt.savefig(f"attention_plot_time_steps.png")



