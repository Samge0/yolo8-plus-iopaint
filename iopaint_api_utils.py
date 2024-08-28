#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author：samge
# date：2024-06-03 14:50
# describe：iopaint的工具类 - 通过api调用封装为类

import os
import requests
import base64

import configs


# IOPaint的服务地址，除了在本项目中执行 python iopaint_server.py 启动iopaint服务外，也可以选择对接单独部署的iopaint服务
IOPAINT_SERVER_HOST = configs.IOPAINT_SERVER_HOST


class InpaintAPI:

    def __init__(self):
        self.api_inpaint = f"{IOPAINT_SERVER_HOST}/api/v1/inpaint"
        self.headers = {
            "Content-Type": "application/json"
        }
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
        try:
            response = requests.post(self.api_inpaint, json=json_body, headers=self.headers, timeout=self.timeout)
        except requests.ConnectionError:
            msg = "\n"
            msg += "=" * 100
            msg += f"\nFailed to connect to the server.please check if the IOPaint service has started properly：{IOPAINT_SERVER_HOST}.\n"
            if '127.0.0.1' in IOPAINT_SERVER_HOST or 'localhost' in IOPAINT_SERVER_HOST:
                msg += "did you forget to execute 'python iopaint_server.py' to start the iopaint service?\n"
            msg += "=" * 100
            raise ValueError(msg)
        except Exception as e:
            raise e
        

        # 检查响应状态码
        if response.status_code == 200:
            # 将返回的二进制图片数据保存到.cache目录
            with open(output_path, "wb") as f:
                f.write(response.content)
            # print(f"图片已保存到 {output_path}")
        else:
            print(f"请求失败，状态码：{response.status_code}")


if __name__ == "__main__":
    # 使用示例
    image_path = f"{configs.images_dir}/test.png"
    mask_path = "your_test_mask_path.png"
    output_path = f"{configs.cache_dir}/output.png"

    # 创建InpaintAPI类的实例 + 发送请求
    inpaint_api = InpaintAPI()
    inpaint_api.send_inpaint_request(image_path, mask_path, output_path)