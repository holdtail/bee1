import tkinter as tk #用于创建图形用户界面
from tkinter import ttk, filedialog, messagebox
import cv2  #用于图像处理和计算机视觉任务
from PIL import Image, ImageTk  #用于图像显示和处理
import os
import json  #用于文件操作和数据序列化
import numpy as np
import matplotlib.pyplot as plt  #matplotlib: 用于创建统计图表
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ultralytics import YOLO


class BeeDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("蜜蜂检测系统")
        self.root.geometry("1000x600")  #设置窗口标题和大小

        # 数据集路径
        self.dataset_path = "D:/Bee/archive"

        # 加载训练好的模型
        self.model = None
        self.load_model()

        # 初始化图像相关属性
        self.current_image = None
        self.current_image_path = None
        self.processed_image = None
        self.original_image = None

        # 数据集统计信息
        self.dataset_stats = self.load_dataset_stats()

        # 创建界面
        self.create_widgets()

    def load_model(self):#  模型加载
        """加载训练好的模型"""
        model_paths = [
            './bee_detection/yolov8_bee_train/weights/best.pt',
            './yolov8_bee_train/weights/best.pt',
            './best.pt',
            './models/yolov8n.pt'  # 添加手动下载的模型路径
        ]

        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    self.model = YOLO(model_path)
                    print(f"模型加载成功: {model_path}")
                    return
                except Exception as e:
                    print(f"加载模型失败 {model_path}: {e}")

        print("未能加载模型，请先训练模型或检查路径")
        messagebox.showwarning("警告", "未能加载模型，请先训练模型")

    def load_dataset_stats(self):
        """加载数据集统计信息"""
        stats = {
            'total_images': 9664,
            'total_boxes': 13402,
            'avg_bees_per_image': 1.4,
            'max_bees_per_image': 11,
            'image_sizes': {
                'avg_pixels': 0.92e6,
                'min_pixels': 0.18e6,
                'max_pixels': 2.59e6
            }
        }

        # 尝试从COCO标注文件中获取实际统计信息
        try:
            for split in ['train', 'val', 'test']:
                coco_json_path = os.path.join(self.dataset_path, split, '_annotations.coco.json')
                if os.path.exists(coco_json_path):
                    with open(coco_json_path, 'r') as f:
                        coco_data = json.load(f)
                    stats['total_images'] = len(coco_data['images'])
                    stats['total_boxes'] = len(coco_data['annotations'])

                    # 计算平均每张图像的蜜蜂数量
                    if stats['total_images'] > 0:
                        stats['avg_bees_per_image'] = stats['total_boxes'] / stats['total_images']

                    break
        except Exception as e:
            print(f"无法从COCO文件获取统计信息: {e}")

        return stats

    def create_widgets(self):
        """创建界面组件"""
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置行列权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # 标题
        title_label = ttk.Label(main_frame, text="蜜蜂检测系统", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # 左侧面板 - 控制和统计信息
        left_panel = ttk.LabelFrame(main_frame, text="控制面板", padding="10")
        left_panel.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.W), padx=(0, 10))
        left_panel.columnconfigure(0, weight=1)

        # 图像处理按钮
        processing_frame = ttk.LabelFrame(left_panel, text="图像处理", padding="10")
        processing_frame.grid(row=0, column=0, pady=5, sticky=(tk.W, tk.E))

        ttk.Button(processing_frame, text="灰度化", command=self.convert_to_grayscale).grid(row=0, column=0, pady=2,
                                                                                            sticky=(tk.W, tk.E))
        ttk.Button(processing_frame, text="边缘检测", command=self.edge_detection).grid(row=0, column=1, pady=2,
                                                                                        sticky=(tk.W, tk.E))
        ttk.Button(processing_frame, text="直方图均衡化", command=self.histogram_equalization).grid(row=1, column=0,
                                                                                                    pady=2,
                                                                                                    sticky=(tk.W, tk.E))
        ttk.Button(processing_frame, text="高斯模糊", command=self.gaussian_blur).grid(row=1, column=1, pady=2,
                                                                                       sticky=(tk.W, tk.E))
        ttk.Button(processing_frame, text="锐化", command=self.sharpen_image).grid(row=2, column=0, pady=2,
                                                                                   sticky=(tk.W, tk.E))
        ttk.Button(processing_frame, text="二值化", command=self.binarize_image).grid(row=2, column=1, pady=2,
                                                                                      sticky=(tk.W, tk.E))
        ttk.Button(processing_frame, text="噪声添加", command=self.add_noise).grid(row=3, column=0, pady=2,
                                                                                   sticky=(tk.W, tk.E))
        ttk.Button(processing_frame, text="形态学操作", command=self.morphological_operations).grid(row=3, column=1,
                                                                                                    pady=2,
                                                                                                    sticky=(tk.W, tk.E))
        ttk.Button(processing_frame, text="重置图像", command=self.reset_image).grid(row=4, column=0, columnspan=2,
                                                                                     pady=5, sticky=(tk.W, tk.E))

        # 检测按钮
        detection_frame = ttk.LabelFrame(left_panel, text="检测功能", padding="10")
        detection_frame.grid(row=1, column=0, pady=5, sticky=(tk.W, tk.E))

        ttk.Button(detection_frame, text="选择单张图像", command=self.select_single_image).grid(row=0, column=0, pady=2,
                                                                                                sticky=(tk.W, tk.E))
        ttk.Button(detection_frame, text="选择多张图像", command=self.select_multiple_images).grid(row=1, column=0,
                                                                                                   pady=2,
                                                                                                   sticky=(tk.W, tk.E))
        ttk.Button(detection_frame, text="显示数据集统计", command=self.show_dataset_stats).grid(row=2, column=0,
                                                                                                 pady=2,
                                                                                                 sticky=(tk.W, tk.E))
        ttk.Button(detection_frame, text="浏览测试集图像", command=self.browse_test_images).grid(row=3, column=0,
                                                                                                 pady=2,
                                                                                                 sticky=(tk.W, tk.E))
        ttk.Button(detection_frame, text="运行检测", command=self.run_detection).grid(row=4, column=0, pady=5,
                                                                                      sticky=(tk.W, tk.E))

        # 统计信息区域
        stats_frame = ttk.LabelFrame(left_panel, text="数据集统计摘要", padding="10")
        stats_frame.grid(row=2, column=0, pady=5, sticky=(tk.W, tk.E))

        stats_text = f"""总图像数: {self.dataset_stats['total_images']}
        总标注框: {self.dataset_stats['total_boxes']}
        平均每图蜜蜂数: {self.dataset_stats['avg_bees_per_image']:.1f}
        单图最多蜜蜂: {self.dataset_stats['max_bees_per_image']}
        平均像素: {self.dataset_stats['image_sizes']['avg_pixels'] / 1e6:.2f}百万"""

        stats_label = ttk.Label(stats_frame, text=stats_text, justify=tk.LEFT)
        stats_label.grid(row=0, column=0, sticky=(tk.W, tk.E))

        # 右侧面板 - 图像显示和结果
        right_panel = ttk.LabelFrame(main_frame, text="图像和结果", padding="10")
        right_panel.grid(row=1, column=1, sticky=(tk.N, tk.S, tk.E, tk.W))
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(0, weight=1)

        # 图像显示区域
        self.image_label = ttk.Label(right_panel)
        self.image_label.grid(row=0, column=0, pady=(0, 10))

        # 结果区域
        result_frame = ttk.LabelFrame(right_panel, text="检测结果", padding="10")
        result_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)

        # 创建文本框显示结果
        self.result_text = tk.Text(result_frame, height=10, width=50)
        result_scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=result_scrollbar.set)

        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        result_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # 配置权重
        left_panel.rowconfigure(2, weight=1)
        right_panel.rowconfigure(0, weight=1)
        right_panel.columnconfigure(0, weight=1)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)

    def convert_to_grayscale(self):
            """灰度化处理"""
            if self.processed_image is None:
                messagebox.showwarning("警告", "请先选择图像")
                return

            gray_image = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
            gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
            self.processed_image = gray_image
            self.display_processed_image()
            self.result_text.insert(tk.END, "已应用灰度化处理\n")

    def edge_detection(self):
            """边缘检测"""
            if self.processed_image is None:
                messagebox.showwarning("警告", "请先选择图像")
                return

            gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            self.processed_image = edges
            self.display_processed_image()
            self.result_text.insert(tk.END, "已应用Canny边缘检测\n")

    def histogram_equalization(self):
            """直方图均衡化"""
            if self.processed_image is None:
                messagebox.showwarning("警告", "请先选择图像")
                return

            # 转换为YUV色彩空间，对Y通道进行均衡化
            yuv = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            equalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            self.processed_image = equalized
            self.display_processed_image()
            self.result_text.insert(tk.END, "已应用直方图均衡化\n")

    def gaussian_blur(self):
            """高斯模糊"""
            if self.processed_image is None:
                messagebox.showwarning("警告", "请先选择图像")
                return

            blurred = cv2.GaussianBlur(self.processed_image, (15, 15), 0)
            self.processed_image = blurred
            self.display_processed_image()
            self.result_text.insert(tk.END, "已应用高斯模糊(15x15)\n")

    def sharpen_image(self):
            """图像锐化"""
            if self.processed_image is None:
                messagebox.showwarning("警告", "请先选择图像")
                return

            kernel = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
            sharpened = cv2.filter2D(self.processed_image, -1, kernel)
            self.processed_image = sharpened
            self.display_processed_image()
            self.result_text.insert(tk.END, "已应用图像锐化\n")

    def binarize_image(self):
            """图像二值化"""
            if self.processed_image is None:
                messagebox.showwarning("警告", "请先选择图像")
                return

            gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            self.processed_image = binary
            self.display_processed_image()
            self.result_text.insert(tk.END, "已应用二值化处理\n")

    def add_noise(self):
            """添加噪声"""
            if self.processed_image is None:
                messagebox.showwarning("警告", "请先选择图像")
                return

            noisy_image = self.processed_image.copy().astype(np.float32)
            noise = np.random.normal(0, 25, self.processed_image.shape).astype(np.float32)
            noisy_image = noisy_image + noise
            noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
            self.processed_image = noisy_image
            self.display_processed_image()
            self.result_text.insert(tk.END, "已添加高斯噪声\n")

    def morphological_operations(self):
            """形态学操作"""
            if self.processed_image is None:
                messagebox.showwarning("警告", "请先选择图像")
                return

            # 创建形态学操作选择窗口
            morph_window = tk.Toplevel(self.root)
            morph_window.title("形态学操作")
            morph_window.geometry("300x200")

            ttk.Label(morph_window, text="选择操作类型:").pack(pady=10)

            operation_var = tk.StringVar(value="dilate")
            ttk.Radiobutton(morph_window, text="膨胀", variable=operation_var, value="dilate").pack()
            ttk.Radiobutton(morph_window, text="腐蚀", variable=operation_var, value="erode").pack()
            ttk.Radiobutton(morph_window, text="开运算", variable=operation_var, value="open").pack()
            ttk.Radiobutton(morph_window, text="闭运算", variable=operation_var, value="close").pack()

            def apply_morphological():
                operation = operation_var.get()
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
                kernel = np.ones((5, 5), np.uint8)

                if operation == "dilate":
                    result = cv2.dilate(gray, kernel, iterations=1)
                elif operation == "erode":
                    result = cv2.erode(gray, kernel, iterations=1)
                elif operation == "open":
                    result = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
                elif operation == "close":
                    result = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

                result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
                self.processed_image = result
                self.display_processed_image()
                self.result_text.insert(tk.END, f"已应用{operation}操作\n")
                morph_window.destroy()

            ttk.Button(morph_window, text="应用", command=apply_morphological).pack(pady=20)

    def reset_image(self):
            """重置图像到原始状态"""
            if self.original_image is not None:
                self.processed_image = self.original_image.copy()
                self.display_processed_image()
                self.result_text.insert(tk.END, "已重置图像\n")

    def display_processed_image(self):
            """显示处理后的图像"""
            if self.processed_image is not None:
                display_image = self.processed_image.copy()
                height, width = display_image.shape[:2]
                max_size = 400

                if height > max_size or width > max_size:
                    scale = max_size / max(height, width)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    display_image = cv2.resize(display_image, (new_width, new_height))

                display_image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(display_image_rgb)
                photo = ImageTk.PhotoImage(image_pil)

                self.image_label.configure(image=photo)
                self.image_label.image = photo

    def run_detection(self):
            """运行检测"""
            if self.processed_image is None and self.current_image is not None:
                self.processed_image = self.current_image.copy()

            if self.processed_image is not None:
                # 将处理后的图像用于检测
                image_rgb = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB)

                try:
                    results = self.model(image_rgb)
                    result_image = results[0].plot()

                    # 显示检测结果
                    height, width = result_image.shape[:2]
                    max_size = 400
                    if height > max_size or width > max_size:
                        scale = max_size / max(height, width)
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        result_image = cv2.resize(result_image, (new_width, new_height))

                    result_image_pil = Image.fromarray(result_image)
                    photo = ImageTk.PhotoImage(result_image_pil)

                    self.image_label.configure(image=photo)
                    self.image_label.image = photo

                    # 显示检测结果文本
                    boxes = results[0].boxes
                    num_detections = len(boxes) if boxes is not None else 0

                    result_str = f"检测到蜜蜂数量: {num_detections}\n\n"
                    if num_detections > 0:
                        result_str += "检测详情:\n"
                        for i, box in enumerate(boxes):
                            conf = box.conf.item()
                            result_str += f"蜜蜂 {i + 1}: 置信度 {conf:.2f}\n"

                    self.result_text.delete(1.0, tk.END)
                    self.result_text.insert(tk.END, result_str)

                except Exception as e:
                    messagebox.showerror("错误", f"检测失败: {e}")

    def select_single_image(self):
        """选择单张图像进行处理"""
        file_path = filedialog.askopenfilename(
            title="选择图像文件",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp")]
        )

        if file_path:
            # 读取图像并设置相关变量
            self.current_image_path = file_path
            self.current_image = cv2.imread(file_path)
            if self.current_image is not None:
                self.processed_image = self.current_image.copy()
                self.original_image = self.current_image.copy()
                self.display_processed_image()  # 显示图像
                self.result_text.insert(tk.END, f"已加载图像: {os.path.basename(file_path)}\n")
            else:
                messagebox.showerror("错误", f"无法读取图像: {file_path}")

    def select_multiple_images(self):
        """选择多张图像进行处理"""
        file_paths = filedialog.askopenfilenames(
            title="选择图像文件",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp")]
        )

        if file_paths:
            for file_path in file_paths:
                self.process_image(file_path)

    def process_image(self, image_path):
        """处理图像并显示结果"""
        if self.model is None:
            messagebox.showerror("错误", "模型未加载，无法进行检测")
            return

        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            messagebox.showerror("错误", f"无法读取图像: {image_path}")
            return

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 进行预测
        try:
            results = self.model(image_rgb)
        except Exception as e:
            messagebox.showerror("错误", f"预测失败: {e}")
            return

        # 绘制检测结果
        result_image = results[0].plot()

        # 显示图像
        self.display_image(result_image)

        # 显示结果文本
        self.display_results(results[0], os.path.basename(image_path))

    def display_image(self, image):
        """在界面上显示图像"""
        # 调整图像大小以适应界面
        height, width = image.shape[:2]
        max_size = 600
        if height > max_size or width > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))

        # 转换为PIL图像并显示
        image_pil = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image_pil)

        self.image_label.configure(image=photo)
        self.image_label.image = photo  # 保持引用

    def display_results(self, result, image_name):
        """显示检测结果"""
        # 获取检测信息
        boxes = result.boxes
        num_detections = len(boxes) if boxes is not None else 0

        # 构建结果文本
        result_str = f"图像: {image_name}\n"
        result_str += f"检测到蜜蜂数量: {num_detections}\n\n"

        if num_detections > 0:
            result_str += "检测详情:\n"
            for i, box in enumerate(boxes):
                conf = box.conf.item()
                result_str += f"蜜蜂 {i + 1}: 置信度 {conf:.2f}\n"

        # 更新文本框
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, result_str)

    def show_dataset_stats(self):
        """显示数据集统计信息的详细图表"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 创建新窗口
        stats_window = tk.Toplevel(self.root)
        stats_window.title("数据集统计信息")
        stats_window.geometry("800x600")

        try:
            # 获取真实的统计信息
            stats = self.calculate_real_dataset_stats()

            # 创建图表
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

            # 1. 蜜蜂数量分布
            bee_count_distribution = stats['bee_count_distribution']
            labels = [f'{count}只' if count < 4 else '4只及以上' for count in bee_count_distribution.keys()]
            counts = list(bee_count_distribution.values())

            ax1.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
            ax1.set_title('每张图像的蜜蜂数量分布')
            ax1.axis('equal')  # 确保饼图是圆形

            # 2. 图像分辨率分布
            resolutions = stats['resolution_distribution']
            res_labels = list(resolutions.keys())
            res_counts = list(resolutions.values())

            ax2.bar(res_labels, res_counts)
            ax2.set_title('图像分辨率分布')
            ax2.set_ylabel('图像数量')
            ax2.tick_params(axis='x', rotation=45)

            # 在柱状图上显示数值
            for i, v in enumerate(res_counts):
                ax2.text(i, v + max(res_counts) * 0.01, str(v), ha='center', va='bottom')

            # 3. 数据集划分
            splits = stats['dataset_split']
            split_labels = list(splits.keys())
            split_counts = list(splits.values())
            colors = ['#66b3ff', '#ffcc99', '#99ff99']

            bars = ax3.bar(split_labels, split_counts, color=colors)
            ax3.set_title('数据集划分')
            ax3.set_ylabel('图像数量')

            # 在柱状图上显示数值和百分比
            total_images = sum(split_counts)
            for i, (bar, count) in enumerate(zip(bars, split_counts)):
                height = bar.get_height()
                percentage = (count / total_images) * 100
                ax3.text(bar.get_x() + bar.get_width() / 2., height + max(split_counts) * 0.01,
                         f'{count}\n({percentage:.1f}%)', ha='center', va='bottom')

            # 4. 蜜蜂大小分布（基于边界框面积）
            size_distribution = stats['bee_size_distribution']
            size_labels = ['小', '中', '大']
            size_counts = [
                size_distribution['small'],
                size_distribution['medium'],
                size_distribution['large']
            ]
            size_colors = ['#ff9999', '#66b3ff', '#99ff99']

            ax4.bar(size_labels, size_counts, color=size_colors)
            ax4.set_title('检测到的蜜蜂大小分布')
            ax4.set_ylabel('蜜蜂数量')

            # 在柱状图上显示数值
            for i, v in enumerate(size_counts):
                ax4.text(i, v + max(size_counts) * 0.01, str(v), ha='center', va='bottom')

            plt.tight_layout()

            # 将图表嵌入到Tkinter窗口中
            canvas = FigureCanvasTkAgg(fig, master=stats_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # 添加统计摘要文本
            summary_text = f"""
            数据集统计摘要:
            - 总图像数: {stats['total_images']}
            - 总蜜蜂数量: {stats['total_bees']}
            - 平均每图蜜蜂数: {stats['avg_bees_per_image']:.2f}
            - 单图最多蜜蜂: {stats['max_bees_per_image']}
            - 最常见的分辨率: {stats['most_common_resolution']}
            - 蜜蜂大小分布: 小({size_counts[0]}), 中({size_counts[1]}), 大({size_counts[2]})
            """

            summary_label = ttk.Label(stats_window, text=summary_text, justify=tk.LEFT)
            summary_label.pack(pady=10)

        except Exception as e:
            messagebox.showerror("错误", f"无法计算统计信息: {e}")
            stats_window.destroy()

    def calculate_real_dataset_stats(self):
        """计算真实的数据集统计信息"""
        stats = {
            'total_images': 0,
            'total_bees': 0,
            'avg_bees_per_image': 0,
            'max_bees_per_image': 0,
            'bee_count_distribution': {},
            'resolution_distribution': {},
            'dataset_split': {'训练集': 0, '验证集': 0, '测试集': 0},
            'bee_size_distribution': {'small': 0, 'medium': 0, 'large': 0},
            'most_common_resolution': ''
        }

        # 定义数据集路径
        splits = ['train', 'val', 'test']

        for split in splits:
            split_dir = os.path.join(self.dataset_path, split)
            annotation_file = os.path.join(split_dir, '_annotations.coco.json')

            if not os.path.exists(annotation_file):
                print(f"警告: 未找到 {split} 集的标注文件: {annotation_file}")
                continue

            try:
                with open(annotation_file, 'r', encoding='utf-8') as f:
                    coco_data = json.load(f)

                # 统计图像信息
                images_info = {img['id']: img for img in coco_data['images']}
                annotations = coco_data['annotations']

                # 按图像分组标注
                image_annotations = {}
                for ann in annotations:
                    image_id = ann['image_id']
                    if image_id not in image_annotations:
                        image_annotations[image_id] = []
                    image_annotations[image_id].append(ann)

                # 更新统计信息
                stats['dataset_split'][f'{split}集'] = len(images_info)
                stats['total_images'] += len(images_info)

                # 统计蜜蜂数量和分辨率
                bee_counts = []
                for img_id, img_info in images_info.items():
                    img_anns = image_annotations.get(img_id, [])
                    bee_count = len(img_anns)
                    bee_counts.append(bee_count)

                    # 更新蜜蜂数量分布
                    if bee_count >= 4:
                        bee_count_key = 4  # 将4只及以上的合并
                    else:
                        bee_count_key = bee_count

                    stats['bee_count_distribution'][bee_count_key] = stats['bee_count_distribution'].get(bee_count_key,
                                                                                                         0) + 1

                    # 统计分辨率
                    resolution = f"{img_info['width']}x{img_info['height']}"
                    stats['resolution_distribution'][resolution] = stats['resolution_distribution'].get(resolution,
                                                                                                        0) + 1

                    # 统计蜜蜂大小
                    for ann in img_anns:
                        stats['total_bees'] += 1
                        bbox = ann['bbox']
                        area = bbox[2] * bbox[3]  # 宽 * 高

                        if area < 1000:  # 小蜜蜂
                            stats['bee_size_distribution']['small'] += 1
                        elif area < 5000:  # 中等蜜蜂
                            stats['bee_size_distribution']['medium'] += 1
                        else:  # 大蜜蜂
                            stats['bee_size_distribution']['large'] += 1

                # 更新最大蜜蜂数量
                if bee_counts:
                    current_max = max(bee_counts)
                    if current_max > stats['max_bees_per_image']:
                        stats['max_bees_per_image'] = current_max

            except Exception as e:
                print(f"处理 {split} 集时出错: {e}")
                continue

        # 计算平均值
        if stats['total_images'] > 0:
            stats['avg_bees_per_image'] = stats['total_bees'] / stats['total_images']

        # 找到最常见的分辨率
        if stats['resolution_distribution']:
            stats['most_common_resolution'] = max(stats['resolution_distribution'].items(), key=lambda x: x[1])[0]

        # 整理蜜蜂数量分布（确保所有可能的值都存在）
        for count in range(0, 5):  # 0-3只，4只及以上
            if count not in stats['bee_count_distribution']:
                stats['bee_count_distribution'][count] = 0

        # 对蜜蜂数量分布进行排序
        stats['bee_count_distribution'] = dict(sorted(stats['bee_count_distribution'].items()))

        return stats

    def browse_test_images(self):
        """浏览测试集图像"""
        test_dir = "D:/Bee/archive/test"
        if not os.path.exists(test_dir):
            messagebox.showerror("错误", f"测试集路径不存在: {test_dir}")
            return

        # 创建浏览窗口
        browse_window = tk.Toplevel(self.root)
        browse_window.title("浏览测试集图像")
        browse_window.geometry("1000x700")

        # 获取测试图像列表
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(image_extensions)]

        if not image_files:
            messagebox.showinfo("信息", "测试集中没有找到图像文件")
            browse_window.destroy()
            return

        # 创建图像选择框架
        select_frame = ttk.Frame(browse_window)
        select_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(select_frame, text="选择图像:").pack(side=tk.LEFT)
        image_var = tk.StringVar()
        image_combo = ttk.Combobox(select_frame, textvariable=image_var, values=image_files, width=50)
        image_combo.pack(side=tk.LEFT, padx=10)
        image_combo.current(0)

        # 图像显示区域
        image_frame = ttk.Frame(browse_window)
        image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        image_label = ttk.Label(image_frame)
        image_label.pack()

        # 结果区域
        result_frame = ttk.LabelFrame(browse_window, text="检测结果", padding="10")
        result_frame.pack(fill=tk.X, padx=10, pady=10)

        result_text = tk.Text(result_frame, height=8, width=50)
        result_scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=result_text.yview)
        result_text.configure(yscrollcommand=result_scrollbar.set)

        result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        result_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        def show_selected_image(event=None):
            """显示选中的图像"""
            selected_image = image_var.get()
            if selected_image:
                image_path = os.path.join(test_dir, selected_image)
                self.process_and_display_image(image_path, image_label, result_text)

        # 绑定选择事件
        image_combo.bind('<<ComboboxSelected>>', show_selected_image)

        # 显示第一张图像
        if image_files:
            show_selected_image()

    def process_and_display_image(self, image_path, image_label, result_text):
        """处理并显示图像"""
        if self.model is None:
            return

        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            result_text.delete(1.0, tk.END)
            result_text.insert(tk.END, f"无法读取图像: {image_path}")
            return

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 进行预测
        try:
            results = self.model(image_rgb)
        except Exception as e:
            result_text.delete(1.0, tk.END)
            result_text.insert(tk.END, f"预测失败: {e}")
            return

        # 绘制检测结果
        result_image = results[0].plot()

        # 显示图像
        height, width = result_image.shape[:2]
        max_size = 500
        if height > max_size or width > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            result_image = cv2.resize(result_image, (new_width, new_height))

        image_pil = Image.fromarray(result_image)
        photo = ImageTk.PhotoImage(image_pil)

        image_label.configure(image=photo)
        image_label.image = photo

        # 显示结果
        boxes = results[0].boxes
        num_detections = len(boxes) if boxes is not None else 0

        result_str = f"检测到蜜蜂数量: {num_detections}\n\n"

        if num_detections > 0:
            result_str += "检测详情:\n"
            for i, box in enumerate(boxes):
                conf = box.conf.item()
                result_str += f"蜜蜂 {i + 1}: 置信度 {conf:.2f}\n"

        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, result_str)


def main():
    """主函数"""
    root = tk.Tk()
    app = BeeDetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()