import os
import yaml
from ultralytics import YOLO
import torch


def check_and_convert_dataset():
    """检查并转换数据集格式"""
    base_path = "D:/Bee/archive"
    yolo_dataset_path = os.path.join(base_path, "bee_yolo_dataset")
    yaml_config_path = os.path.join(yolo_dataset_path, "bee_dataset.yaml")

    # 如果YOLO格式数据集不存在，则进行转换
    if not os.path.exists(yaml_config_path):
        print("YOLO格式数据集不存在，开始转换...")
        try:
            # 动态导入转换函数
            from coco_to_yolo import convert_all_splits
            yolo_dataset_path = convert_all_splits(base_path)
            yaml_config_path = os.path.join(yolo_dataset_path, "bee_dataset.yaml")
        except Exception as e:
            print(f"数据集转换失败: {e}")
            return None

    return yaml_config_path


def train_model():
    """训练YOLOv8模型"""
    # 检查并转换数据集
    yaml_config_path = check_and_convert_dataset()
    if not yaml_config_path:
        print("无法获取数据集配置，请检查数据集路径")
        return

    # 检查是否有可用的GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 加载预训练模型
    model = YOLO('yolov8n.pt')  # 使用nano版本

    # 训练模型
    results = model.train(
        data=yaml_config_path,
        epochs=20,
        imgsz=416,
        batch=16,
        device=device,
        project='bee_detection',
        name='yolov8_bee_train',
        save=True,
        exist_ok=True,
        patience=10,  # 早停耐心值
        lr0=0.01,  # 初始学习率
        lrf=0.01,  # 最终学习率
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,  # 边界框损失权重
        cls=0.5,  # 分类损失权重
        dfl=1.5,  # 分布焦点损失权重
    )

    # 验证模型
    print("开始验证模型...")
    metrics = model.val()
    print(f"验证结果: mAP50-95: {metrics.box.map:.3f}, mAP50: {metrics.box.map50:.3f}")
    # mAP50：IoU阈值为0.5时的平均精度
    # mAP50-95：IoU阈值从0.5到0.95的平均精度

    # 导出最佳模型
    best_model_path = './bee_detection/yolov8_bee_train/weights/best.pt'
    if os.path.exists(best_model_path):
        best_model = YOLO(best_model_path)
        # 导出为ONNX格式（可选）
        best_model.export(format='onnx')
        print(f"最佳模型已导出: {best_model_path}")
    else:
        print("警告: 未找到最佳模型文件")

    print("训练完成！")


if __name__ == '__main__':
    train_model()