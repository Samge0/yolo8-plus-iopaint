## docker of yolo8-plus-iopaint-dev


### run .devcontainer/devcontainer.json
the `Dev Containers` mode of vscode, refer to[vscode_devcontainer docs](https://code.visualstudio.com/docs/devcontainers/create-dev-container)
- install the `Dev Containers` plugin for vscode
- shortcut key 'Ctrl+Shift+P' to open the command panel, select `Remote-Containers: Reopen in Container`
- automatically build the container and wait for completion to access the workspace inside the container


### optional steps
- build docker
    ```shell
    docker build . -t samge/yolo8-plus-iopaint-dev-base -f .devcontainer/Dockerfile-dev-base --build-arg PROXY=http://192.168.50.48:7890
    ```

- upload
    ```shell
    docker push samge/yolo8-plus-iopaint-dev-base
    ```


### other instructions
- CUDA Toolkit and Minimum Required Driver Version for CUDA Minor Version Compatibility
Minimum Required Driver Version for CUDA Minor Version Compatibility*

    | CUDA Toolkit Version | Linux x86_64 Driver Version | Windows x86_64 Driver Version |
    |----------------------|-----------------------------|-------------------------------|
    | CUDA 12.x            | >= 525.60.13                | >= 528.33                     |
    | CUDA 11.1.x - 11.8.x | >= 450.80.02                | >= 452.39                     |
    | CUDA 11.0 (11.0.3)   | >= 450.36.06**              | >= 451.22**                   |

    **Note: Ensure that the driver version is at least the specified minimum version to maintain compatibility with the corresponding CUDA toolkit version. [click here to view the official nvidia documentation>>](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#title-resolved-issues)

- If you encounter the following types of errors, it may be due to issues with the NVIDIA graphics card driver. Try reinstalling the graphics card driver and then restart the system. If you are using the latest version of the graphics card driver, consider downgrading to an earlier version and then try again.
    ```text
    xxx/site-packages/torch/cuda/__init__.py:141: UserWarning: CUDA initialization: Unexpected error from cudaGetDeviceCount(). Did you run some cuda functions before calling NumCudaDevices() that might have already set an error? Error 500: named symbol not found (Triggered internally at ../c10/cuda/CUDAFunctions.cpp:108.)`
    ```