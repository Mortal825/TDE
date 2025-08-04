import os

def count_images_in_folder(folder_path):
    # 定义常见的图片扩展名
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')
    count = 0

    # 遍历文件夹中的所有文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(image_extensions):  # 检查文件扩展名
                count += 1

    return count

# 示例用法
folder_path = "/media/haichao/af0987fa-e592-4d73-8a57-4056cb7bebf0/haichao/datasets/voc2012/VOCdevkit/VOC2012/JPEGImages"
print(f"图片数量: {count_images_in_folder(folder_path)}")
