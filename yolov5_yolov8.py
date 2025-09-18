import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import seaborn as sns
import time
import warnings
from tqdm import tqdm
import json
import requests
from tqdm import tqdm

warnings.filterwarnings('ignore')


class BeeDetectionComparator:
    def __init__(self, data_path, weights_path):
        """
        初始化比较器

        参数:
            data_path: 数据集路径
            weights_path: 预训练权重路径
        """
        self.data_path = Path(data_path)
        self.weights_path = Path(weights_path)

        # 尝试导入YOLOv8
        try:
            from ultralytics import YOLO
            self.YOLO = YOLO
        except ImportError:
            raise ImportError("请安装ultralytics包: pip install ultralytics")

        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

    def check_model_files(self):
        """检查所有可用的模型文件"""
        print("检查模型文件...")

        # 所有可能的模型文件
        all_models = [
            # YOLOv5 预训练模型
            self.weights_path / "yolov5s.pt",
            self.weights_path / "yolov5m.pt",
            # YOLOv5 自定义训练模型
            self.weights_path / "custom_yolov5s.pt",
            self.weights_path / "custom_yolov5m.pt",
            # YOLOv8 预训练模型
            self.weights_path / "yolov8n.pt",
            self.weights_path / "yolov8s.pt",
            self.weights_path / "yolov8m.pt",
            # YOLOv8 蜜蜂检测训练模型
            self.weights_path.parent / "bee_detection" / "yolov8_bee_train" / "weights" / "best.pt"
        ]

        available_models = []
        for model_path in all_models:
            if model_path.exists():
                available_models.append(model_path)
                print(f"✓ 找到模型: {model_path}")
            else:
                print(f"✗ 未找到: {model_path}")

        return available_models

    def load_models(self):
        """加载所有可用的YOLOv5和YOLOv8模型"""
        print("=" * 50)
        print("检查并加载模型...")

        # 先检查所有模型文件
        available_models = self.check_model_files()

        # 加载YOLOv5模型
        print("\n加载YOLOv5模型...")
        yolov5_models = {}

        # 优先加载自定义训练模型
        custom_models = [
            self.weights_path / "custom_yolov5s.pt",
            self.weights_path / "custom_yolov5m.pt",
            self.weights_path / "yolov5s.pt",
            self.weights_path / "yolov5m.pt"
        ]

        for model_path in custom_models:
            if model_path.exists():
                model_name = model_path.stem
                print(f"尝试加载: {model_name}")
                model = self.load_yolov5_model(model_path)
                if model is not None:
                    yolov5_models[model_name] = model
                    print(f"✓ {model_name} 加载成功")
                    break

        self.yolov5_models = yolov5_models

        # 加载YOLOv8模型
        print("\n加载YOLOv8模型...")
        yolov8_models = {}

        # 优先加载蜜蜂检测训练模型
        bee_model_path = self.weights_path / "bee_detection" / "yolov8_bee_train" / "weights" / "best.pt"
        if bee_model_path.exists():
            print("尝试加载: YOLOv8蜜蜂检测模型")
            model = self.load_yolov8_model(bee_model_path)
            if model is not None:
                yolov8_models["yolov8_bee"] = model
                print("✓ YOLOv8蜜蜂检测模型加载成功")

        # 加载其他YOLOv8预训练模型
        yolov8_pretrained = [
            self.weights_path / "yolov8s.pt",
            self.weights_path / "yolov8m.pt",
            self.weights_path / "yolov8n.pt"
        ]

        for model_path in yolov8_pretrained:
            if model_path.exists():
                model_name = model_path.stem
                print(f"尝试加载: {model_name}")
                model = self.load_yolov8_model(model_path)
                if model is not None:
                    yolov8_models[model_name] = model
                    print(f"✓ {model_name} 加载成功")

        self.yolov8_models = yolov8_models

        print(f"\n模型加载完成!")
        print(f"已加载 YOLOv5 模型: {list(self.yolov5_models.keys())}")
        print(f"已加载 YOLOv8 模型: {list(self.yolov8_models.keys())}")

    def load_yolov5_model(self, weight_path):
        """加载YOLOv5模型 - 增强错误处理"""
        try:
            # 清除torch.hub缓存以避免版本冲突
            torch.hub.set_dir(str(self.weights_path / "hub_cache"))

            # 使用torch.hub加载YOLOv5
            model = torch.hub.load('ultralytics/yolov5', 'custom',
                                   path=str(weight_path),
                                   force_reload=True,
                                   trust_repo=True)
            model.to(self.device)
            model.conf = 0.25  # 设置置信度阈值
            return model
        except Exception as e:
            print(f"加载YOLOv5模型错误: {e}")
            return None

    def load_yolov8_model(self, weight_path):
        """加载YOLOv8模型 - 处理PyTorch 2.6兼容性"""
        try:
            # 方法1: 正常加载
            model = self.YOLO(str(weight_path))
            return model
        except Exception as e:
            print(f"方法1加载YOLOv8模型错误: {e}")
            try:
                # 方法2: 处理PyTorch 2.6的weights_only问题
                from ultralytics.nn.tasks import DetectionModel
                import torch.serialization

                # 添加安全全局变量
                torch.serialization.add_safe_globals([DetectionModel])

                # 重新尝试加载
                model = self.YOLO(str(weight_path))
                return model
            except Exception as e2:
                print(f"方法2加载YOLOv8模型错误: {e2}")
                return None

    def prepare_data(self):
        """准备测试数据 - 适配新的数据集结构"""
        test_images = []

        # 检查测试集目录结构
        test_dirs = [
            self.data_path / 'test' / 'images',
            self.data_path / 'val' / 'images',  # 也检查验证集
            self.data_path / 'train' / 'images'  # 也检查训练集
        ]

        for test_dir in test_dirs:
            if test_dir.exists():
                print(f"在 {test_dir} 中查找测试图像...")
                # 查找图像文件
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                    images = list(test_dir.glob(ext))
                    test_images.extend(images)
                    if images:
                        print(f"找到 {len(images)} 个 {ext} 文件")

        # 如果没有找到测试图像，尝试其他可能的目录结构
        if not test_images:
            print("在标准目录中未找到图像，尝试其他位置...")
            # 直接在数据集根目录下查找
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                images = list(self.data_path.glob(ext))
                test_images.extend(images)
                if images:
                    print(f"在根目录找到 {len(images)} 个 {ext} 文件")

        if not test_images:
            print("警告: 未找到测试图像")
        else:
            print(f"总共找到 {len(test_images)} 个测试图像")

        return test_images

    def yolov5_detect(self, img_path, model_name=None, conf_thres=0.25):
        """使用指定的YOLOv5模型进行检测"""
        if not self.yolov5_models:
            print("没有可用的YOLOv5模型，跳过检测")
            return [], 0, "no_model"

        # 选择模型
        if model_name is None:
            # 使用第一个可用的模型
            model_name = list(self.yolov5_models.keys())[0]

        if model_name not in self.yolov5_models:
            print(f"未找到YOLOv5模型: {model_name}")
            return [], 0, model_name

        model = self.yolov5_models[model_name]

        try:
            # 使用YOLOv5进行推理
            start_time = time.time()
            results = model(img_path)
            inference_time = time.time() - start_time

            # 处理检测结果
            detections = []
            if hasattr(results, 'xyxy'):
                # 处理单个图像的结果
                for result in results.xyxy[0]:
                    if len(result) >= 6:  # 确保有足够的元素
                        x1, y1, x2, y2, conf, cls = result[:6]
                        if conf >= conf_thres:  # 应用置信度阈值
                            detections.append({
                                'bbox': [x1.item(), y1.item(), x2.item(), y2.item()],
                                'confidence': conf.item(),
                                'class': int(cls.item())
                            })
            elif hasattr(results, 'pandas'):
                # 处理pandas格式的结果
                df = results.pandas().xyxy[0]
                for _, row in df.iterrows():
                    if row['confidence'] >= conf_thres:
                        detections.append({
                            'bbox': [row['xmin'], row['ymin'], row['xmax'], row['ymax']],
                            'confidence': row['confidence'],
                            'class': row['class']
                        })

            print(f"YOLOv5({model_name})检测到 {len(detections)} 个目标")
            return detections, inference_time, model_name
        except Exception as e:
            print(f"YOLOv5检测错误: {e}")
            return [], 0, model_name

    def yolov8_detect(self, img_path, model_name=None, conf_thres=0.25):
        """使用指定的YOLOv8模型进行检测"""
        if not self.yolov8_models:
            print("没有可用的YOLOv8模型，跳过检测")
            return [], 0, "no_model"

        # 选择模型
        if model_name is None:
            # 使用第一个可用的模型
            model_name = list(self.yolov8_models.keys())[0]

        if model_name not in self.yolov8_models:
            print(f"未找到YOLOv8模型: {model_name}")
            return [], 0, model_name

        model = self.yolov8_models[model_name]

        try:
            start_time = time.time()
            results = model(img_path, conf=conf_thres, verbose=False)
            inference_time = time.time() - start_time

            detections = []
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for box in result.boxes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = box.cls[0].cpu().numpy()

                        if conf >= conf_thres:
                            detections.append({
                                'bbox': xyxy.tolist(),
                                'confidence': float(conf),
                                'class': int(cls)
                            })

            print(f"YOLOv8({model_name})检测到 {len(detections)} 个目标")
            return detections, inference_time, model_name
        except Exception as e:
            print(f"YOLOv8检测错误: {e}")
            return [], 0, model_name

    def find_label_file(self, img_path):
        """根据图像路径找到对应的标注文件"""
        # 尝试在不同目录结构中查找标注文件
        possible_label_dirs = [
            img_path.parent.parent / 'labels' / img_path.parent.name,
            img_path.parent.parent / 'labels',
            img_path.parent.with_name('labels'),
            img_path.parent  # 标注文件可能在同一个目录
        ]

        # 生成可能的标注文件名（保持相同的文件名，扩展名为.txt）
        label_filename = img_path.stem + '.txt'

        for label_dir in possible_label_dirs:
            label_path = label_dir / label_filename
            if label_path.exists():
                return label_path

        return None

    def load_ground_truth(self, img_path):
        """加载真实标注 - 适配YOLO格式"""
        ground_truth = []

        # 查找标注文件
        label_path = self.find_label_file(img_path)

        if label_path and label_path.exists():
            try:
                print(f"加载标注文件: {label_path}")
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])

                            # 读取图像获取尺寸
                            img = cv2.imread(str(img_path))
                            if img is not None:
                                h, w = img.shape[:2]

                                # 将YOLO格式转换为绝对坐标
                                x_min = (x_center - width / 2) * w
                                y_min = (y_center - height / 2) * h
                                x_max = (x_center + width / 2) * w
                                y_max = (y_center + height / 2) * h

                                ground_truth.append({
                                    'bbox': [x_min, y_min, x_max, y_max],
                                    'class': cls_id
                                })
            except Exception as e:
                print(f"加载标注文件错误: {e}")
        else:
            print(f"警告: 未找到 {img_path.name} 的标注文件")

        print(f"加载了 {len(ground_truth)} 个真实标注")
        return ground_truth

    def calculate_metrics(self, detections, ground_truth, iou_threshold=0.5):
        """计算评估指标"""
        tp, fp, fn = 0, 0, 0

        # 将检测结果与真实标注匹配
        matched_gt = set()
        detections.sort(key=lambda x: x['confidence'], reverse=True)

        for det in detections:
            matched = False
            for i, gt in enumerate(ground_truth):
                if i in matched_gt:
                    continue

                iou = self.calculate_iou(det['bbox'], gt['bbox'])
                if iou >= iou_threshold:
                    tp += 1
                    matched_gt.add(i)
                    matched = True
                    break

            if not matched:
                fp += 1

        fn = len(ground_truth) - len(matched_gt)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }

    def calculate_iou(self, box1, box2):
        """计算两个边界框的IoU"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # 计算交集区域
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

        # 计算并集区域
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def run_comparison(self, test_images, sample_size=10):
        """运行比较实验 - 测试所有模型"""
        results = []

        if not test_images:
            print("没有测试数据可用")
            return pd.DataFrame(results)

        # 获取所有可用的模型名称
        yolov5_model_names = list(self.yolov5_models.keys())
        yolov8_model_names = list(self.yolov8_models.keys())

        if not yolov5_model_names and not yolov8_model_names:
            print("没有可用的模型进行测试")
            return pd.DataFrame(results)

        for i, img_path in enumerate(tqdm(test_images[:sample_size], desc="处理图像")):
            print(f"\n处理图像 {i + 1}/{min(sample_size, len(test_images))}: {img_path.name}")

            # 检查图像文件是否存在
            if not img_path.exists():
                print(f"警告: 图像文件不存在: {img_path}")
                continue

            # 加载真实标注
            ground_truth = self.load_ground_truth(img_path)
            print(f"真实标注数量: {len(ground_truth)}")

            # 测试所有YOLOv5模型
            for model_name in yolov5_model_names:
                yolov5_detections, yolov5_time, used_model = self.yolov5_detect(str(img_path), model_name)
                yolov5_metrics = self.calculate_metrics(yolov5_detections, ground_truth) if yolov5_detections else {
                    'precision': 0, 'recall': 0, 'f1_score': 0, 'tp': 0, 'fp': 0, 'fn': len(ground_truth)}

                results.append({
                    'image': img_path.name,
                    'model_type': 'YOLOv5',
                    'model_name': used_model,
                    'detections': len(yolov5_detections),
                    'inference_time': yolov5_time,
                    'precision': yolov5_metrics['precision'],
                    'recall': yolov5_metrics['recall'],
                    'f1_score': yolov5_metrics['f1_score'],
                    'tp': yolov5_metrics['tp'],
                    'fp': yolov5_metrics['fp'],
                    'fn': yolov5_metrics['fn'],
                    'gt_count': len(ground_truth)
                })

            # 测试所有YOLOv8模型
            for model_name in yolov8_model_names:
                yolov8_detections, yolov8_time, used_model = self.yolov8_detect(str(img_path), model_name)
                yolov8_metrics = self.calculate_metrics(yolov8_detections, ground_truth) if yolov8_detections else {
                    'precision': 0, 'recall': 0, 'f1_score': 0, 'tp': 0, 'fp': 0, 'fn': len(ground_truth)}

                results.append({
                    'image': img_path.name,
                    'model_type': 'YOLOv8',
                    'model_name': used_model,
                    'detections': len(yolov8_detections),
                    'inference_time': yolov8_time,
                    'precision': yolov8_metrics['precision'],
                    'recall': yolov8_metrics['recall'],
                    'f1_score': yolov8_metrics['f1_score'],
                    'tp': yolov8_metrics['tp'],
                    'fp': yolov8_metrics['fp'],
                    'fn': yolov8_metrics['fn'],
                    'gt_count': len(ground_truth)
                })

        return pd.DataFrame(results)

    def visualize_results(self, results_df, output_dir='comparison_results'):
        """可视化比较结果"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # 设置绘图风格
        plt.style.use('default')
        sns.set_palette("viridis")

        if results_df.empty:
            print("没有结果数据可可视化")
            return

        # 按模型类型和名称分组
        model_groups = results_df.groupby(['model_type', 'model_name']).mean(numeric_only=True).reset_index()

        # 绘制性能指标比较
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('模型性能比较', fontsize=16)

        # F1分数比较
        sns.barplot(data=model_groups, x='model_name', y='f1_score', hue='model_type', ax=axes[0, 0])
        axes[0, 0].set_title('F1分数比较')
        axes[0, 0].set_ylabel('F1 Score')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # 精确度比较
        sns.barplot(data=model_groups, x='model_name', y='precision', hue='model_type', ax=axes[0, 1])
        axes[0, 1].set_title('精确度比较')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # 召回率比较
        sns.barplot(data=model_groups, x='model_name', y='recall', hue='model_type', ax=axes[1, 0])
        axes[1, 0].set_title('召回率比较')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # 推理时间比较
        sns.barplot(data=model_groups, x='model_name', y='inference_time', hue='model_type', ax=axes[1, 1])
        axes[1, 1].set_title('推理时间比较')
        axes[1, 1].set_ylabel('Inference Time (s)')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 保存详细结果
        results_df.to_csv(output_dir / 'detailed_results.csv', index=False)

        # 生成模型总结
        summary_df = results_df.groupby(['model_type', 'model_name']).agg({
            'precision': 'mean',
            'recall': 'mean',
            'f1_score': 'mean',
            'inference_time': 'mean',
            'detections': 'sum',
            'tp': 'sum',
            'fp': 'sum',
            'fn': 'sum'
        }).round(4)

        summary_df.to_csv(output_dir / 'model_summary.csv')

        # 生成总结报告
        self.generate_summary_report(results_df, output_dir / 'summary_report.txt')

        print(f"所有结果已保存到 {output_dir} 目录")

    def generate_summary_report(self, results_df, report_path):
        """生成详细的总结报告"""
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("蜜蜂检测模型比较报告\n")
            f.write("=" * 60 + "\n\n")

            if results_df.empty:
                f.write("没有可用的比较结果数据\n")
                return

            # 按模型分组统计
            model_stats = results_df.groupby(['model_type', 'model_name']).agg({
                'precision': 'mean',
                'recall': 'mean',
                'f1_score': 'mean',
                'inference_time': 'mean',
                'tp': 'sum',
                'fp': 'sum',
                'fn': 'sum'
            }).round(4)

            f.write("模型性能汇总:\n")
            f.write("-" * 60 + "\n")
            for (model_type, model_name), stats in model_stats.iterrows():
                f.write(f"\n{model_type} - {model_name}:\n")
                f.write(f"  平均精确度: {stats['precision']:.4f}\n")
                f.write(f"  平均召回率: {stats['recall']:.4f}\n")
                f.write(f"  平均F1分数: {stats['f1_score']:.4f}\n")
                f.write(f"  平均推理时间: {stats['inference_time']:.4f}s\n")
                f.write(f"  总TP: {stats['tp']}, FP: {stats['fp']}, FN: {stats['fn']}\n")

            # 找出最佳模型
            best_f1 = model_stats['f1_score'].idxmax()
            best_precision = model_stats['precision'].idxmax()
            best_recall = model_stats['recall'].idxmax()
            fastest = model_stats['inference_time'].idxmin()

            f.write("\n最佳模型:\n")
            f.write("-" * 60 + "\n")
            f.write(f"最佳F1分数: {best_f1[1]} ({best_f1[0]}) - {model_stats.loc[best_f1, 'f1_score']:.4f}\n")
            f.write(
                f"最佳精确度: {best_precision[1]} ({best_precision[0]}) - {model_stats.loc[best_precision, 'precision']:.4f}\n")
            f.write(f"最佳召回率: {best_recall[1]} ({best_recall[0]}) - {model_stats.loc[best_recall, 'recall']:.4f}\n")
            f.write(f"最快推理: {fastest[1]} ({fastest[0]}) - {model_stats.loc[fastest, 'inference_time']:.4f}s\n")

            f.write(f"\n测试图像数量: {len(results_df['image'].unique())}\n")
            f.write(f"总真实标注数: {results_df['gt_count'].sum()}\n")


# 使用示例
if __name__ == "__main__":
    # 设置路径
    DATA_PATH = r"D:\Bee\yolo_format"
    WEIGHTS_PATH = r"C:\Users\10835\PycharmProjects\Bee1\models"

    # 初始化比较器
    comparator = BeeDetectionComparator(DATA_PATH, WEIGHTS_PATH)

    # 加载模型
    comparator.load_models()

    # 准备数据
    test_images = comparator.prepare_data()

    # 运行比较实验
    print("开始比较实验...")
    results_df = comparator.run_comparison(test_images, sample_size=10)

    # 可视化结果并生成报告
    comparator.visualize_results(results_df, output_dir='comparison_results')

    print("比较实验完成! 结果保存在 'comparison_results' 目录中")