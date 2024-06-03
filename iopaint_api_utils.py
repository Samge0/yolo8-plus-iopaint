#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author：samge
# date：2024-06-03 14:50
# describe：iopaint的工具类 - 通过api调用封装为类

import os
import requests
import base64

class InpaintAPI:

    def __init__(self):
        self.api_inpaint = "http://127.0.0.1:8000/api/v1/inpaint"
        self.headers = {"Content-Type": "application/json" }
        self.timeout = 30

    def convert_image_to_base64(self, image_path):
        """将图片文件转换为base64字符串"""
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_string

    def send_inpaint_request(self, image_path, mask_path, output_path):
        """发送POST请求到inpaint API，并保存返回的图片"""

        # 保证输出路径存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 将图片和标记转换为base64字符串
        image_base64 = self.convert_image_to_base64(image_path)
        mark_base64 = self.convert_image_to_base64(mask_path)

        # 构建请求的JSON body
        json_body = {
            "image": image_base64,
            "mask": mark_base64
        }

        # 发送POST请求
        response = requests.post(self.api_inpaint, json=json_body, headers=self.headers, timeout=self.timeout)

        # 检查响应状态码
        if response.status_code == 200:
            # 将返回的二进制图片数据保存到.cache目录
            with open(output_path, "wb") as f:
                f.write(response.content)
            print(f"图片已保存到 {output_path}")
        else:
            print(f"请求失败，状态码：{response.status_code}")


if __name__ == "__main__":
    # 使用示例
    image_path = "images/test.png"
    mask_path = "your_test_mask_path.png"
    output_path = ".cache/output.png"

    # 创建InpaintAPI类的实例 + 发送请求
    inpaint_api = InpaintAPI()
    inpaint_api.send_inpaint_request(image_path, mask_path, output_path)