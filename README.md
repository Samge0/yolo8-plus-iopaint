## 移除品牌logo水印的demo

本demo使用[ultralytics-YOLO8](https://github.com/ultralytics/ultralytics)对水印位置进行模型训练&检测，然后使用[IOPaint](https://github.com/Sanster/IOPaint)移除检测到的水印。

本demo中使用的[last.pt](models/last.pt)模型来自[yolo8-watermark-brand](https://github.com/Samge0/yolo8-watermark-brand)仓库。


### 当前开发环境使用的关键依赖版本
```text
python==3.8.18
torch==2.3.0+cu118
torchvision==0.18.0+cu118
ultralytics==8.2.26
IOPaint==1.3.3
```


### 安装依赖
- 【二选一】安装torch-cpu版
    ```shell
    pip install torch torchvision
    ```
- 【二选一】安装torch-cuda版
    ```shell
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    ```
- 【必要】安装依赖
    ```shell
    pip install -r requirements.txt
    ```


### 运行
```shell
python remove_watermark.py
```
