#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author：samge
# date：2024-06-03 14:06
# describe：使用onnx推理

import os
import cv2
import torch

from onnx_utils import OnnxUtils
from iopaint_utils import IOPaintCmdUtil, IOPaintApiUtil

CUDA_IS_AVAILABLE = torch.cuda.is_available()

output_dir = ".cache"                               # 输出目录
model_path = "models/last.onnx"                     # yolo模型路径，pt转onnx模型可参考`yolo_utils.py`的mian函数
device = "cuda" if CUDA_IS_AVAILABLE else "cpu"     # 设备类型

SAVE_ONNX_BORDER_IMAGE = False                      # 是否保存onnx检测到边框的结果

USE_IOPAINT_API = True                              # 【推荐】是否使用iopaint的api方式去除水印，如果设置为True，需要先运行iopaint服务：python iopaint_server.py 或使用自定义的IOPaint服务


# 擦除水印
def detect_and_erase(image_path, model_path, output_dir, device="cpu"):
    # 初始化ONNX模型和IOPaint工具
    onnx_obj = OnnxUtils(model_path, conf_thres=0.75, iou_thres=0.75, imgsz=[288, 288])
    iopaint_obj = IOPaintApiUtil(device=device) if USE_IOPAINT_API else IOPaintCmdUtil(device=device)

    # 读取图像
    image = cv2.imread(image_path)

    # 使用YOLO模型获取边界框
    bboxes, scores, class_ids = onnx_obj(image)
    print(bboxes, scores, class_ids)

    # 创建并保存掩码图像
    mask = iopaint_obj.create_mask(image, bboxes)
    iopaint_obj.erase_watermark(image_path, mask, output_dir)

    # 【可选】绘制onnx检测到的目标边框并保存
    if SAVE_ONNX_BORDER_IMAGE:
        onnx_obj.draw_boxes(image, bboxes, scores, class_ids)
        output_path = f"{output_dir}/border_{os.path.basename(image_path)}"
        cv2.imwrite(output_path, image)


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
    """
    使用示例
    """
    
    if USE_IOPAINT_API:
        print("=====【温馨提示】使用iopaint的api方式去除水印，如果设置为True，需要先运行iopaint服务：python iopaint_server.py 或使用自定义的IOPaint服务=====\n")

    os.makedirs(output_dir, exist_ok=True)

    # 移除单张水印
    image_path = "images/test.png"
    detect_and_erase(image_path, model_path, output_dir, device=device)

    # 移除某个目录所有图片水印
    _test_batch(batch_dir="images")

    print("all done")
