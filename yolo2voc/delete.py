import os
import cv2
# 定义照片文件夹和标注文件夹路径
photos_dir = '/media/haichao/af0987fa-e592-4d73-8a57-4056cb7bebf0/haichao/datasets/voc2012/images/val2012'        # 照片文件夹路径
annotations_dir = '/media/haichao/af0987fa-e592-4d73-8a57-4056cb7bebf0/haichao/datasets/voc2012/labels/val2012'  # 标注文件夹路径
output_dir = './output'       # 输出文件夹路径

# 创建输出文件夹（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 获取照片文件名（不带后缀）
photo_files = {os.path.splitext(photo)[0] for photo in os.listdir(photos_dir) if os.path.isfile(os.path.join(photos_dir, photo))}

# 获取标注文件名（不带后缀）
annotation_files = {os.path.splitext(annotation)[0] for annotation in os.listdir(annotations_dir) if os.path.isfile(os.path.join(annotations_dir, annotation))}

# 找出多余的标注文件
extra_annotations = annotation_files - photo_files

# 删除多余的标注文件
for extra in extra_annotations:
    file_path = os.path.join(annotations_dir, f"{extra}.txt")
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted: {file_path}")

# 检查是否数量相等
if len(photo_files) == len(annotation_files - extra_annotations):
    print("删除多余标注后，数量相等！")
else:
    print(f"数量仍然不相等！照片文件数量：{len(photo_files)}，标注文件数量：{len(annotation_files - extra_annotations)}")
    exit()

# 检查是否一一对应
mismatched_files = photo_files - annotation_files
if mismatched_files:
    print(f"以下照片没有对应的标注文件：{mismatched_files}")
    exit()
else:
    print("照片和标注文件一一对应！")

# 绘制标注在照片上
for photo_file in photo_files:
    photo_path = os.path.join(photos_dir, f"{photo_file}.jpg")  # 假设图片后缀是 .jpg
    annotation_path = os.path.join(annotations_dir, f"{photo_file}.txt")
    output_path = os.path.join(output_dir, f"{photo_file}_annotated.jpg")
    
    # 读取图片
    image = cv2.imread(photo_path)
    if image is None:
        print(f"无法读取图片：{photo_path}")
        continue
    height, width, _ = image.shape

    # 读取YOLO标注文件
    with open(annotation_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        # YOLO格式：class x_center y_center width height
        parts = line.strip().split()
        if len(parts) != 5:
            print(f"无效的标注格式：{line}")
            continue
        cls, x_center, y_center, box_width, box_height = map(float, parts)

        # 转换为图像坐标
        x_center *= width
        y_center *= height
        box_width *= width
        box_height *= height

        x_min = int(x_center - box_width / 2)
        y_min = int(y_center - box_height / 2)
        x_max = int(x_center + box_width / 2)
        y_max = int(y_center + box_height / 2)

        # 绘制矩形框
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        # 显示类别
        cv2.putText(image, f"Class {int(cls)}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 保存绘制后的图片
    cv2.imwrite(output_path, image)
    # print(f"保存标注后的图片：{output_path}")

print("所有标注绘制完成！")
