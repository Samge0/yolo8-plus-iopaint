## 移除品牌logo水印的Demo

本Demo使用[Ultralytics-YOLO8](https://github.com/ultralytics/ultralytics)对水印位置进行检测，然后使用[IOPaint](https://github.com/Sanster/IOPaint)移除yolo识别的目标水印。

本Demo使用的[last.pt](models/last.pt)模型来自[yolo8-watermark-brand](https://github.com/Samge0/yolo8-watermark-brand)仓库。


### 使用IOPaint的api方式
本Demo支持使用`IOPaint`的`api方式`去除水印，只需在[configs.py](configs.py)中配置`USE_IOPAINT_API=True`，可减少批量操作时iopaint命令行方式的初始化耗时。

如果配置`USE_IOPAINT_API=True`，需要先启动iopaint服务：
```shell
python iopaint_server.py
```

当然，也可以选择对接单独部署的`IOPaint`服务，只需要在[configs.py](configs.py)中配置自定义的`IOPAINT_SERVER_HOST`即可。


### 当前开发环境使用的关键依赖版本
<details> <summary>点击展开查看依赖信息</summary>

```text
python==3.8.18

torch==2.3.0+cu118
torchvision==0.18.0+cu118
ultralytics==8.2.26
IOPaint==1.3.3
onnxruntime_gpu==1.18.0

# the onnx dependency is to automatically export the onnx model at train time
onnx==1.16.1
onnx-simplifier==0.4.36
onnxsim==0.4.36
onnxslim==0.1.28
```

</details>


### 本地调试
1、【推荐】使用vscode的`Dev Containers`模式，参考[.devcontainer/README.md](.devcontainer/README.md)

2、【可选】其他虚拟环境方式
- 用conda创建一个env环境：
    ```shell
    conda create -n yolo8-plus-iopaint python=3.8.18 -y
    ```

- 进入env环境
    ```shell    
    conda activate yolo8-plus-iopaint
    ```

- 安装依赖
    - 【二选一】安装`torch-cpu`版
        ```shell
        pip install torch torchvision
        ```

    - 【二选一】安装`torch-cuda`版
        ```shell
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
        ```

    - 【必要】安装依赖
        ```shell
        pip install -r requirements.txt
        ```


### 运行Demo：
- Demo1：`ultralytics + IOPaint（命令行方式）`，[脚本：remove_watermark.py](remove_watermark.py)
    ```shell
    python remove_watermark.py
    ```

- Demo2：`onnxruntime + IOPaint（命令行方式）`，[脚本：remove_watermark_with_onnx.py](remove_watermark_with_onnx.py)<br>pt转onnx模型可参考[yolo_utils.py](yolo_utils.py)的mian函数
    ```shell
    python remove_watermark_with_onnx.py
    ```


### 错误处理[可选]
<details> <summary>点击展开查看</summary>
> 1、如果遇到`Could not locate zlibwapi.dll. Please make sure it is in your library path`错误，需要下载相关dll放置到目标位置：

- [点击下载：https://pan.baidu.com/s/1SrxZFkxwpwydn1fuFaWtgw?pwd=6cgb 提取码: 6cgb](https://pan.baidu.com/s/1SrxZFkxwpwydn1fuFaWtgw?pwd=6cgb)
- lib文件放到`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.x\lib` （这里以windows系统为例，其中`v11.x`是实际安装的cuda版本路径）
- dll文件放到`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.x\bin` （这里以windows系统为例，其中`v11.x`是实际安装的cuda版本路径）
</details>


### 相关截图

|before|after|
|:--------:|:--------:|
|![before](https://github.com/Samge0/yolo8-plus-iopaint/assets/17336101/801bdcef-88d7-449d-a48a-428e117b58ab)|![after](https://github.com/Samge0/yolo8-plus-iopaint/assets/17336101/a465b913-4aa1-4c04-a12b-c0211d47b6bc)|