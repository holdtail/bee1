import requests
import os
from pathlib import Path
from tqdm import tqdm


def download_yolov5_model(model_size='s', save_dir='models'):
    """
    下载YOLOv5预训练模型

    参数:
        model_size: 模型大小 ('n', 's', 'm', 'l', 'x')
        save_dir: 保存目录
    """
    # 模型URL映射
    model_urls = {
        'n': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt',
        's': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt',
        'm': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt',
        'l': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt',
        'x': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x.pt'
    }

    if model_size not in model_urls:
        print(f"错误: 不支持的模型大小 '{model_size}'，请选择: n, s, m, l, x")
        return None

    url = model_urls[model_size]
    filename = f"yolov5{model_size}.pt"
    save_path = Path(save_dir) / filename

    # 如果文件已存在，跳过下载
    if save_path.exists():
        print(f"模型已存在: {save_path}")
        return save_path

    # 创建保存目录
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"正在下载 YOLOv5{model_size} 模型...")
    print(f"下载地址: {url}")

    try:
        # 使用流式下载并显示进度条
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 检查请求是否成功

        # 获取文件大小
        total_size = int(response.headers.get('content-length', 0))

        # 下载文件
        with open(save_path, 'wb') as file, tqdm(
                desc=filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)

        print(f"✓ 下载完成: {save_path}")
        print(f"文件大小: {save_path.stat().st_size / (1024 * 1024):.2f} MB")

        return save_path

    except requests.exceptions.RequestException as e:
        print(f"下载失败: {e}")
        # 清理可能下载失败的文件
        if save_path.exists():
            save_path.unlink()
        return None


# 使用示例
if __name__ == "__main__":
    # 下载YOLOv5s模型（推荐）
    model_path = download_yolov5_model('s', 'models')

    if model_path:
        print(f"模型已保存到: {model_path}")
    else:
        print("下载失败，请检查网络连接")