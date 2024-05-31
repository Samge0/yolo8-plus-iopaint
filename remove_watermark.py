#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author：samge
# date：2024-05-31 16:04
# describe：

"""
使用yolo8+iopaint结合，用yolo8识别目标水印位置，调用iopaint移除水印
"""

import uuid
import cv2
import numpy as np
import os
import subprocess
from ultralytics import YOLO
from ultralytics.utils import checks


CUDA_IS_AVAILABLE = checks.cuda_is_available()

output_dir = ".cache"                               # 输出目录
model_path = "models/last.pt"                      # yolo模型路径
device = "cuda" if CUDA_IS_AVAILABLE else "cpu"     # 设备类型

# yolo模型对象
_model = None


# 价值yolo模型
def load_yolo_model(model_path):
    global _model
    if not _model:
        _model = YOLO(model_path)
    return _model


# 擦除水印
def detect_and_erase(image_path, model_path, output_dir, device="cpu"):
    # 加载训练好的YOLOv8模型
    model = load_yolo_model(model_path)

    # 读取图像
    image = cv2.imread(image_path)
    image_name = os.path.basename(image_path)

    # 使用模型进行检测
    results = model(image, conf=0.75)  # filter values with confidence level > 0.75

    # 获取目标的边界框坐标
    bboxes = results[0].boxes[0].cpu().data.numpy()

    # 创建一个与图像大小相同的掩码
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # 填充检测到的目标区域为白色
    padding = 1  # 【可选】目标位置的上下左右的扩充距离，避免在yolo识别的框比较小时，iopaint会因mark蒙版没有完全覆盖水印导致遗留少量阴影的情况
    height, width = image.shape[:2]
    for bbox in bboxes:
        x1, y1, x2, y2, conf, cls = map(int, bbox)
        x1 = np.clip(x1 - padding, 0, width)
        y1 = np.clip(y1 - padding, 0, height)
        x2 = np.clip(x2 + padding, 0, width)
        y2 = np.clip(y2 + padding, 0, height)
        mask[y1:y2, x1:x2] = 255

    # 创建保存结果的文件夹（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{image_name}"

    # 保存临时掩码图像：https://www.iopaint.com/batch_process
    temp_mask_path = f'.cache/{uuid.uuid4()}{image_name}'
    cv2.imwrite(temp_mask_path, mask)

    # 调用iopaint的终端命令进行擦除
    command = [
        'iopaint', 'run',
        '--model=lama',
        f'--device={device}',
        f'--image={image_path}',
        f'--mask={temp_mask_path}',
        f'--output={output_dir}'
    ]

    subprocess.run(command, check=True)

    # 移除临时文件
    os.remove(temp_mask_path) if os.path.exists(temp_mask_path) else None

    print(f"水印已移除： {image_path} => {output_path}")


# Run batch
def _test_batch(batch_dir: str):
    file_list = os.listdir(batch_dir)
    total_size = len(file_list)
    for i in range(total_size):
        filename = file_list[i]

        is_goal_image = filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")
        if not is_goal_image:
            continue

        image_path = os.path.join(batch_dir, filename).replace(os.sep, "/")
        print(f"【{i+1}/{total_size}】 is running: {image_path} => {output_dir}")
        detect_and_erase(image_path, model_path, output_dir, device=device)


if __name__ == "__main__":
    # 使用示例

    # 移除单张水印
    image_path = "images/test.png"
    detect_and_erase(image_path, model_path, output_dir, device=device)

    # 移除某个目录所有图片水印
    _test_batch(batch_dir="images")

    print("all done")
