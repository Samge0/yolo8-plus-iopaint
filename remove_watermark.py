#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author：samge
# date：2024-05-31 16:04
# describe：使用yolo8+iopaint结合，用yolo8识别目标水印位置，调用iopaint移除水印

import cv2
import os
import configs
from yolo_utils import YOLOUtils
from iopaint_utils import IOPaintCmdUtil, IOPaintApiUtil


output_dir = configs.cache_dir                      # 输出目录
model_path = f"{configs.models_dir}/last.pt"        # yolo模型路径
device = configs.device                             # 设备类型

USE_IOPAINT_API = configs.USE_IOPAINT_API           #【推荐】是否使用iopaint的api方式去除水印，如果设置为True，需要先运行iopaint服务：python iopaint_server.py 或使用自定义的IOPaint服务


# 擦除水印
def detect_and_erase(image_path, model_path, output_dir, device="cpu"):
    # 初始化YOLO模型和IOPaint工具
    yolo_obj = YOLOUtils(model_path)
    iopaint_obj = IOPaintApiUtil(device=device) if USE_IOPAINT_API else IOPaintCmdUtil(device=device)

    # 读取图像
    image = cv2.imread(image_path)

    # 使用YOLO模型获取边界框
    bboxes = yolo_obj.get_bboxes(image)

    # 创建并保存掩码图像
    mask = iopaint_obj.create_mask(image, bboxes)
    iopaint_obj.erase_watermark(image_path, mask, output_dir)


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
        print(f"\n【{i+1}/{total_size}】 running: ")
        detect_and_erase(image_path, model_path, output_dir, device=device)


if __name__ == "__main__":
    """
    使用示例
    """

    if USE_IOPAINT_API:
        print("=====【温馨提示】使用iopaint的api方式去除水印，如果设置为True，需要先运行iopaint服务：python iopaint_server.py 或使用自定义的IOPaint服务=====\n")

    os.makedirs(output_dir, exist_ok=True)

    # 移除单张水印
    image_path = f"{configs.images_dir}/test.png"
    detect_and_erase(image_path, model_path, output_dir, device=device)

    # 移除某个目录所有图片水印
    _test_batch(batch_dir=configs.images_dir)

    print("\nall done")
