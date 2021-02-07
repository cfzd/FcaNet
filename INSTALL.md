1. First clone the project and install requirments
    ```
    git clone https://github.com/cfzd/FcaNet.git
    cd FcaNet
    pip install -r requirements.txt
    ```
2. Install Nvidia DALI

    For CUDA 10 
    ```
    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda100
    ```

    For CUDA 11
    ```
    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110
    ```

3. Install Nvidia APEX (Optional, only for mixed precision training)
    ```
    git clone https://github.com/NVIDIA/apex
    cd apex
    pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
    ```

4. Install mmdetection as [https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md) (Optional, only for detection and instance segmentation models).