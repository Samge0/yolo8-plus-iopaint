#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author：samge
# date：2024-06-03 18:25
# describe：iopaint的api服务，如果选择使用api方式调用iopaint，则需要先运行该服务

import subprocess

def start_iopaint_server():
    """
    运行iopaint服务
    """
    model = "lama"
    device = "cuda"
    port = "8000"
    enable_interactive_seg = "--enable-interactive-seg"
    interactive_seg_device = "cuda"
    
    # 构建命令
    command = [
        "iopaint", "start",
        "--model={}".format(model),
        "--device={}".format(device),
        "--port={}".format(port),
        enable_interactive_seg,
        "--interactive-seg-device={}".format(interactive_seg_device)
    ]
    
    # 运行命令
    process = subprocess.Popen(command)
    
    # 等待进程结束
    process.wait()


if __name__ == "__main__":
    start_iopaint_server()
