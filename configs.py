#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author：samge
# date：2024-08-28 11:22
# describe：文件相关类，存在需要在非项目根路径执行脚本的情况，将相关目录设置为绝对路径


import os

import torch

# 项目目录
project_dir = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, '/')

# 缓存目录
cache_dir = f"{project_dir}/.cache"

# 模型目录
models_dir = f"{project_dir}/models"

# 测试图片目录
images_dir = f"{project_dir}/images"

#【推荐】是否使用iopaint的api方式去除水印，如果设置为True，需要先运行iopaint服务：python iopaint_server.py 或使用自定义的IOPaint服务
USE_IOPAINT_API = True                              

# IOPaint的服务地址，除了在本项目中执行 python iopaint_server.py 启动iopaint服务外，也可以选择对接单独部署的iopaint服务
IOPAINT_SERVER_HOST = "http://127.0.0.1:8000"

# 是否cuda
CUDA_IS_AVAILABLE = torch.cuda.is_available()

# 设备类型
device = "cuda" if CUDA_IS_AVAILABLE else "cpu"      
