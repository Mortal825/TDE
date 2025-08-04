import pandas as pd
import ast

# 读取 Excel 文件
file1 = "firing_rates.xlsx"  # 表1
file2 = "model_summary.xlsx"  # 表2

df1 = pd.read_excel(file1)  # 读取表1
df2 = pd.read_excel(file2)  # 读取表2
T = 4
# 识别表2中的 mem_update 组索引
mem_update_indices = df2[df2["Layer"].str.contains("mem_update", na=False)].index.tolist()

# 存储每个 mem_update 之间的 Conv2d 组合
conv_groups = []
kernel_groups = []
in_channels_groups = []
out_channels_groups = []
max_kernel_groups = []
groups_list = []  # 记录分组数（groups）

# 遍历 mem_update 组
for i in range(len(mem_update_indices) - 1):
    start, end = mem_update_indices[i], mem_update_indices[i + 1]

    # 获取 start 和 end 之间的子 DataFrame
    subset = df2.iloc[start:end]

    # 筛选包含 "Conv2d" 的层
    conv_subset = subset[subset["Layer"].str.contains("Conv2d", na=False)]

    # 如果没有找到 Conv2d，跳过当前循环
    if conv_subset.empty:
        continue

    # 提取相关信息
    conv_layers = conv_subset["Layer"].tolist()
    Kernel_Sizes = conv_subset["Kernel Size"].tolist()
    In_Channels = conv_subset["In Channels"].iloc[0]  # 第一个输入通道
    Out_Channels = conv_subset["Out Channels"].iloc[-1]  # 最后一个输出通道
    Groups = list(map(int, conv_subset["Groups"].tolist()))  # 记录分组数，并转换为整数类型

    # 转换 Kernel Size 数据
    Kernel_Sizes = [ast.literal_eval(k) if isinstance(k, str) else k for k in Kernel_Sizes]

    # 找到最大的 Kernel Size（按面积计算）
    max_kernel_size = max(Kernel_Sizes, key=lambda x: x[0] * x[1])

    # 处理分组数：
    # - 如果所有分组数相同，则记录该值
    # - 如果存在不同的分组数，则不记录（设为 "N/A"）
    unique_groups = set(Groups)
    final_groups = str(unique_groups.pop()) if len(unique_groups) == 1 else "N/A"

    # 作为整体信息存入对应的列表
    conv_groups.append(", ".join(conv_layers))
    in_channels_groups.append(str(In_Channels))
    out_channels_groups.append(str(Out_Channels))
    kernel_groups.append(", ".join(map(str, Kernel_Sizes)))
    max_kernel_groups.append(str(max_kernel_size))
    groups_list.append(final_groups)  # 记录唯一分组数或 "N/A"

# 在表1中新增列，并填入找到的信息
df1["Conv"] = pd.Series(conv_groups)  
df1["Kernel Size"] = pd.Series(max_kernel_groups)
df1["In Channels"] = pd.Series(in_channels_groups)
df1["Out Channels"] = pd.Series(out_channels_groups)
df1["Groups"] = pd.Series(groups_list)  # 添加分组信息

# 计算能耗 (Firing Rate × In Channels × Out Channels × (Kernel Size)^2 × 特征图^2)
energy_groups = []

for rate, in_ch, out_ch, kernel, outshape, groups in zip(
        df1["Firing Rate"], df1["In Channels"], df1["Out Channels"],
        df1["Kernel Size"], df1["Tensor Shape"], df1["Groups"]):
    try:
        # 转换 In Channels 和 Out Channels 为整数
        in_ch = int(float(in_ch))  # 将字符串转换为 float，再转换为 int
        out_ch = int(float(out_ch))  # 将字符串转换为 float，再转换为 int
        kernel_size = ast.literal_eval(kernel) if isinstance(kernel, str) else kernel  # 确保转换
        kernel_area = kernel_size[0] * kernel_size[1]  # 计算卷积核面积
        outshape = ast.literal_eval(outshape)
        outmap = outshape[1] * outshape[2]  # 计算特征图面积

        # 处理 Groups
        if isinstance(groups, str) and groups.isdigit():  # 如果 groups 是字符串数字
            groups = int(groups)
        elif isinstance(groups, (int, float)):  # 如果 groups 已经是数值类型
            groups = int(groups)
        else:
            groups = 1  # 设为默认值 1（表示标准卷积）

        # 计算能耗
        energy = rate * in_ch * out_ch * kernel_area * outmap * 0.9 * 10**(-9) * T

        # 如果分组数大于 1，则除以分组数
        if groups > 1:
            energy /= groups

        energy = round(energy, 5)  # 保留 5 位小数
        energy_groups.append(energy)

    except Exception as e:
        print(f"Error calculating energy: {e}")
        energy_groups.append(None)

# 添加能耗列
df1["Energy"] = pd.Series(energy_groups)

# 计算能耗总和并添加到 DataFrame 末尾
energy_sum = df1["Energy"].sum()
print(f"Total energy consumption:{energy_sum}mj")
summary_row = pd.DataFrame({
    "Conv": ["Total"],
    "Kernel Size": ["-"],
    "In Channels": ["-"],
    "Out Channels": ["-"],
    "Energy": [energy_sum]
})

df1 = pd.concat([df1, summary_row], ignore_index=True)

# 保存结果
output_file = "output.xlsx"
df1.to_excel(output_file, index=False)
print(f"处理完成，结果已保存到 {output_file}")
