import os
import shutil

# 数据集路径
voc_root = "/media/haichao/af0987fa-e592-4d73-8a57-4056cb7bebf0/haichao/datasets/voc2012/VOCdevkit/VOC2012"  # 替换为你的 VOC 数据集根目录
images_dir = os.path.join(voc_root, "JPEGImages")
train_txt = os.path.join(voc_root, "ImageSets", "Main", "train.txt")
val_txt = os.path.join(voc_root, "ImageSets", "Main", "val.txt")
# test_txt = os.path.join(voc_root, "ImageSets", "Main", "test.txt")

root = "/media/haichao/af0987fa-e592-4d73-8a57-4056cb7bebf0/haichao/datasets/voc2012/images"
# 输出目录
train_output_dir = os.path.join(root, "train2012")
val_output_dir = os.path.join(root, "val2012")
# test_output_dir = os.path.join(root, "test2007")

# 创建输出目录
os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(val_output_dir, exist_ok=True)
# os.makedirs(test_output_dir, exist_ok=True)
# 按照 txt 文件中的列表复制图片
def copy_images(txt_file, output_dir):
    with open(txt_file, "r") as f:
        image_ids = [line.strip() for line in f.readlines()]
    for image_id in image_ids:
        src_image_path = os.path.join(images_dir, f"{image_id}.jpg")
        dst_image_path = os.path.join(output_dir, f"{image_id}.jpg")
        if os.path.exists(src_image_path):
            shutil.copy(src_image_path, dst_image_path)
            print(f"Copied {src_image_path} to {dst_image_path}")
        else:
            print(f"Image {src_image_path} does not exist!")

# 复制 train.txt 中的图片到 train2007
print("Processing train.txt...")
copy_images(train_txt, train_output_dir)

# 复制 val.txt 中的图片到 val2007
print("Processing val.txt...")
copy_images(val_txt, val_output_dir)

# 复制 test.txt 中的图片到 test2007
# print("Processing val.txt...")
# copy_images(test_txt, test_output_dir)


print("Processing complete!")
