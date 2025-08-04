def read_image_names_from_files(file_paths):
    """从给定的txt文件路径列表中读取图片名字集合"""
    image_names = set()
    for file_path in file_paths:
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    image_names.add(line.strip())
        except FileNotFoundError:
            print(f"文件未找到: {file_path}")
        except Exception as e:
            print(f"读取文件 {file_path} 时发生错误: {e}")
    return image_names

def main():
    # # 定义a群和b群的txt文件路径数组
    # object = ["/media/haichao/af0987fa-e592-4d73-8a57-4056cb7bebf0/haichao/datasets/voc2012/VOCdevkit/VOC2012/ImageSets/Main/train.txt", 
    #            "/media/haichao/af0987fa-e592-4d73-8a57-4056cb7bebf0/haichao/datasets/voc2012/VOCdevkit/VOC2012/ImageSets/Main/val.txt",
    #            ]  # 替换为实际路径
    # segment = ["/media/haichao/af0987fa-e592-4d73-8a57-4056cb7bebf0/haichao/datasets/voc2012/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt", 
    #           "/media/haichao/af0987fa-e592-4d73-8a57-4056cb7bebf0/haichao/datasets/voc2012/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt"]  # 替换为实际路径

    object = ["/media/haichao/af0987fa-e592-4d73-8a57-4056cb7bebf0/haichao/datasets/voc2012/VOCdevkit/VOC2007/ImageSets/Main/train.txt", 
               "/media/haichao/af0987fa-e592-4d73-8a57-4056cb7bebf0/haichao/datasets/voc2012/VOCdevkit/VOC2007/ImageSets/Main/val.txt",
               "/media/haichao/af0987fa-e592-4d73-8a57-4056cb7bebf0/haichao/datasets/voc2012/VOCdevkit/VOC2007/ImageSets/Main/test.txt",
               ]  # 替换为实际路径
    segment = ["/media/haichao/af0987fa-e592-4d73-8a57-4056cb7bebf0/haichao/datasets/voc2012/VOCdevkit/VOC2007/ImageSets/Segmentation/train.txt", 
              "/media/haichao/af0987fa-e592-4d73-8a57-4056cb7bebf0/haichao/datasets/voc2012/VOCdevkit/VOC2007/ImageSets/Segmentation/val.txt",
              "/media/haichao/af0987fa-e592-4d73-8a57-4056cb7bebf0/haichao/datasets/voc2012/VOCdevkit/VOC2007/ImageSets/Segmentation/test.txt"
              ]  # 替换为实际路径


    # 读取两组文件中的图片名字集合
    images_a = read_image_names_from_files(segment)
    images_b = read_image_names_from_files(object)

    # 统计
    common_images = images_a & images_b  # 共有图片名字
    only_in_a = images_a - images_b      # a群有但b群没有的图片名字
    only_in_b = images_b - images_a      # b群有但a群没有的图片名字

    # 输出结果
    print(f"segment和object共有图片名字数量: {len(common_images)}")
    print(f"segment有但object没有的图片名字数量: {len(only_in_a)}")
    print(f"object有但segment没有的图片名字数量: {len(only_in_b)}")

if __name__ == "__main__":
    main()
