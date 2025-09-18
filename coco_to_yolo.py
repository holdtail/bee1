import os
import json
import shutil
from PIL import Image
import yaml


def convert_coco_to_yolo(coco_json_path, images_dir, output_dir, class_names=None):
    """
    将COCO格式的标注转换为YOLO格式

    Args:
        coco_json_path: COCO标注文件路径
        images_dir: 图像文件目录
        output_dir: 输出目录
        class_names: 类别名称列表
    """
    # 最后生成D:/Bee/archive/bee_yolo_dataset/
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    labels_dir = os.path.join(output_dir, 'labels')
    images_output_dir = os.path.join(output_dir, 'images')
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_output_dir, exist_ok=True)

    # 读取COCO标注文件
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # 创建类别映射
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    if class_names is None:
        class_names = list(categories.values())
    else:
        # 确保class_names包含所有类别
        existing_classes = list(categories.values())
        for cls in existing_classes:
            if cls not in class_names:
                class_names.append(cls)

    # 创建图像ID到文件名的映射
    image_id_to_info = {img['id']: img for img in coco_data['images']}

    # 按图像分组标注
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)

    # 处理每个图像
    processed_count = 0
    for image_id, annotations in annotations_by_image.items():
        if image_id not in image_id_to_info:
            continue

        image_info = image_id_to_info[image_id]
        image_file_name = image_info['file_name']
        image_width = image_info['width']
        image_height = image_info['height']

        # 复制图像文件
        src_image_path = os.path.join(images_dir, image_file_name)
        dst_image_path = os.path.join(images_output_dir, image_file_name)
        if os.path.exists(src_image_path):
            shutil.copy2(src_image_path, dst_image_path)
        else:
            print(f"警告: 图像文件不存在: {src_image_path}")
            continue

        # 创建YOLO格式的标注文件
        label_file_name = os.path.splitext(image_file_name)[0] + '.txt'
        label_file_path = os.path.join(labels_dir, label_file_name)

        with open(label_file_path, 'w') as f:
            for ann in annotations:
                # 获取类别ID
                category_id = ann['category_id']
                class_name = categories[category_id]
                class_id = class_names.index(class_name)

                # 获取边界框坐标 (COCO格式: [x, y, width, height])
                bbox = ann['bbox']
                x_center = (bbox[0] + bbox[2] / 2) / image_width
                y_center = (bbox[1] + bbox[3] / 2) / image_height
                width = bbox[2] / image_width
                height = bbox[3] / image_height

                # 确保坐标在[0,1]范围内
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))

                # 写入YOLO格式: class_id x_center y_center width height
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        processed_count += 1

    print(f"转换完成！输出目录: {output_dir}")
    print(f"共处理 {processed_count} 张图像")

    return class_names


def create_yolo_dataset_structure(base_path):
    """
    创建YOLO格式的数据集结构
    """
    # 定义数据集路径
    dataset_path = os.path.join(base_path, 'bee_yolo_dataset')

    # 创建目录结构
    directories = {
        'train': ['images', 'labels'],
        'val': ['images', 'labels'],
        'test': ['images', 'labels']
    }

    for split, subdirs in directories.items():
        for subdir in subdirs:
            os.makedirs(os.path.join(dataset_path, split, subdir), exist_ok=True)

    return dataset_path


def convert_all_splits(base_path):
    """
    转换所有数据分割（训练集、验证集、测试集）
    """
    # 创建YOLO数据集结构
    yolo_dataset_path = create_yolo_dataset_structure(base_path)

    # 定义各分割的路径
    splits = {
        'train': {
            'coco_json': os.path.join(base_path, 'train', '_annotations.coco.json'),
            'images_dir': os.path.join(base_path, 'train')
        },
        'val': {
            'coco_json': os.path.join(base_path, 'val', '_annotations.coco.json'),
            'images_dir': os.path.join(base_path, 'val')
        },
        'test': {
            'coco_json': os.path.join(base_path, 'test', '_annotations.coco.json'),
            'images_dir': os.path.join(base_path, 'test')
        }
    }

    class_names = None

    # 转换每个分割
    for split, paths in splits.items():
        if os.path.exists(paths['coco_json']):
            print(f"正在转换 {split} 集...")
            output_dir = os.path.join(yolo_dataset_path, split)
            class_names = convert_coco_to_yolo(
                paths['coco_json'],
                paths['images_dir'],
                output_dir,
                class_names
            )
        else:
            print(f"警告: 未找到 {split} 集的COCO标注文件: {paths['coco_json']}")

    # 创建YOLO数据集配置文件
    create_yolo_config(yolo_dataset_path, class_names)

    return yolo_dataset_path


def create_yolo_config(dataset_path, class_names):
    """
    创建YOLO数据集配置文件
    """
    config = {
        'path': os.path.abspath(dataset_path),
        'train': 'train',
        'val': 'val',
        'test': 'test',
        'names': {i: name for i, name in enumerate(class_names)},
        'nc': len(class_names)
    }

    config_path = os.path.join(dataset_path, 'bee_dataset.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"配置文件已创建: {config_path}")

    # 打印配置文件内容
    print("配置文件内容:")
    with open(config_path, 'r') as f:
        print(f.read())

    return config_path


if __name__ == "__main__":
    base_path = "D:/Bee/archive"

    if not os.path.exists(base_path):
        print(f"错误: 数据集路径不存在: {base_path}")
        print("请确保数据集路径正确，或者修改 base_path 变量")
    else:
        print("开始转换COCO格式到YOLO格式...")
        yolo_dataset_path = convert_all_splits(base_path)
        print(f"YOLO格式数据集已创建在: {yolo_dataset_path}")

        # 检查生成的文件
        print("\n检查生成的文件结构:")
        for root, dirs, files in os.walk(yolo_dataset_path):
            level = root.replace(yolo_dataset_path, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                if file.endswith('.txt') or file.endswith('.yaml'):
                    print(f"{subindent}{file}")