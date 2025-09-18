import os
import requests
import urllib.request
from pathlib import Path


def download_yolov8_model():
    """手动下载YOLOv8预训练模型"""
    model_urls = {
        'yolov8n.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt',
        'yolov8s.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt',
        'yolov8m.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt'
    }

    # 创建模型目录
    model_dir = Path('./models')
    model_dir.mkdir(exist_ok=True)

    for model_name, url in model_urls.items():
        model_path = model_dir / model_name
        if not model_path.exists():
            print(f"正在下载 {model_name}...")
            try:
                # 尝试使用urllib下载
                urllib.request.urlretrieve(url, model_path)
                print(f"成功下载 {model_name}")
            except Exception as e:
                print(f"下载失败 {model_name}: {e}")
                # 尝试使用requests作为备选方案
                try:
                    response = requests.get(url, stream=True, timeout=30)
                    if response.status_code == 200:
                        with open(model_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        print(f"成功下载 {model_name}")
                    else:
                        print(f"下载失败，HTTP状态码: {response.status_code}")
                except Exception as e2:
                    print(f"所有下载方法都失败: {e2}")
        else:
            print(f"{model_name} 已存在")

    return model_dir / 'yolov8n.pt'


if __name__ == '__main__':
    download_yolov8_model()