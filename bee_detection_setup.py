import os
import requests
import shutil
from pathlib import Path


class BeeDetectionSetup:
    def __init__(self):
        self.base_dir = os.getcwd()
        self.file_urls = {
            '.gitignore': 'https://raw.githubusercontent.com/AjayJohnAlex/Bee_Detection/main/.gitignore',
            'LICENSE': 'https://raw.githubusercontent.com/AjayJohnAlex/Bee_Detection/main/LICENSE',
            'README.md': 'https://raw.githubusercontent.com/AjayJohnAlex/Bee_Detection/main/README.md',
            'app.py': 'https://raw.githubusercontent.com/AjayJohnAlex/Bee_Detection/main/app.py',
            'requirements.txt': 'https://raw.githubusercontent.com/AjayJohnAlex/Bee_Detection/main/requirements.txt',
            'models/custom_yolov5m.pt': 'https://github.com/AjayJohnAlex/Bee_Detection/raw/main/models/custom_yolov5m.pt',
            'models/custom_yolov5s.pt': 'https://github.com/AjayJohnAlex/Bee_Detection/raw/main/models/custom_yolov5s.pt',
            'templates/about_project.html': 'https://raw.githubusercontent.com/AjayJohnAlex/Bee_Detection/main/templates/about_project.html',
            'templates/index.html': 'https://raw.githubusercontent.com/AjayJohnAlex/Bee_Detection/main/templates/index.html',
            'static/style.css': 'https://github.com/AjayJohnAlex/Bee_Detection/blob/main/templates/static/style.css',
            'upload_data/honey-bees-96fps.mp4': 'https://github.com/AjayJohnAlex/Bee_Detection/raw/main/upload_data/honey-bees-96fps.mp4',
            'upload_data/honey_bee_video1.mp4': 'https://github.com/AjayJohnAlex/Bee_Detection/raw/main/upload_data/honey_bee_video1.mp4'
        }

    def create_directory_structure(self):
        """创建必要的目录结构"""
        directories = [
            'models',
            'templates',
            'static',
            'upload_data',
            'results/batch1',
            '__pycache__'
        ]

        for directory in directories:
            os.makedirs(os.path.join(self.base_dir, directory), exist_ok=True)
            print(f"创建目录: {directory}")

    def download_file(self, url, filepath):
        """下载单个文件"""
        try:
            print(f"正在下载: {filepath}")

            # 对于大文件（如模型文件和视频），使用流式下载
            if any(ext in filepath for ext in ['.pt', '.mp4']):
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(filepath, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                else:
                    print(f"下载失败: {filepath}, 状态码: {response.status_code}")
                    return False
            else:
                # 对于小文件，直接下载
                response = requests.get(url)
                if response.status_code == 200:
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                else:
                    print(f"下载失败: {filepath}, 状态码: {response.status_code}")
                    return False

            print(f"成功下载: {filepath}")
            return True

        except Exception as e:
            print(f"下载 {filepath} 时出错: {str(e)}")
            return False

    def download_all_files(self):
        """下载所有文件"""
        success_count = 0
        total_count = len(self.file_urls)

        for relative_path, url in self.file_urls.items():
            filepath = os.path.join(self.base_dir, relative_path)

            # 确保目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            if self.download_file(url, filepath):
                success_count += 1

        print(f"\n下载完成: {success_count}/{total_count} 个文件成功下载")

        if success_count < total_count:
            print("警告: 部分文件下载失败，可能需要手动下载")

    def install_requirements(self):
        """安装所需的Python包"""
        requirements_file = os.path.join(self.base_dir, 'requirements.txt')
        if os.path.exists(requirements_file):
            print("正在安装所需的Python包...")
            os.system(f"pip install -r {requirements_file}")
        else:
            print("requirements.txt 文件未找到，跳过依赖安装")

    def setup_complete(self):
        """完成设置后的提示信息"""
        print("\n" + "=" * 50)
        print("Bee Detection 项目设置完成!")
        print("=" * 50)
        print("\n下一步:")
        print("1. 运行 'python app.py' 启动应用程序")
        print("2. 在浏览器中访问 http://localhost:5000")
        print("\n注意:")
        print("- 确保已安装Python 3.8或更高版本")
        print("- 确保已安装PyTorch (如果需要GPU支持，请安装CUDA版本的PyTorch)")
        print("\n项目结构:")
        for root, dirs, files in os.walk(self.base_dir):
            level = root.replace(self.base_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")

    def run(self):
        """运行完整的设置过程"""
        print("开始设置 Bee Detection 项目...")
        print(f"项目将安装在: {self.base_dir}")

        # 创建目录结构
        self.create_directory_structure()

        # 下载所有文件
        self.download_all_files()

        # 安装依赖
        self.install_requirements()

        # 显示完成信息
        self.setup_complete()


if __name__ == "__main__":
    setup = BeeDetectionSetup()
    setup.run()