from pkg_resources import parse_version
from setuptools import setup, find_packages
import os
assert parse_version(setuptools.__version__)>=parse_version('36.2')

# 读取 README.md 作为长描述
with open(os.path.join(os.path.dirname(__file__), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='DeepRUOT',                        # 包名称
    version='0.1.0',                        # 初始版本号，可根据后续更新进行修改
    author='Zhenyi Zhang',                      # 作者姓名
    author_email='zhenyizhang@stu.pku.edu.cn',  # 作者邮箱
    description='DeepRUOT',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/zhenyiizhang/DeepRUOT',  # 你的项目主页或 GitHub 地址
    packages=find_packages(),               # 自动查找 deepruot 目录下的所有模块
    include_package_data=True,
    install_requires=[
        "torch>=1.11.0",
        "matplotlib",
        "numpy<2",
        "torchdiffeq",
        "torchsde",
        "scipy",
        "scikit-learn",
        "pot",
        "phate",
        "pyyaml",
        "tqdm",
        "seaborn>=0.12.2",
        "pandas",
        "ipywidgets",
        "scanpy",
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',      # 开发状态，根据实际情况修改
        'Intended Audience :: Developers',
        'Natural Language :: Chinese',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # 如果你选择 MIT 许可证
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # 支持的最低 Python 版本
    entry_points={
        # 如果希望在命令行安装后能调用脚本，可配置 console_scripts
        # 示例：
        # 'console_scripts': ['deepruot=deepruot.cli:main'],
    },
)