# Style Transfer Video Temporal Consistency for Webcam

The following proyect is a merge from Learning Blind Video Temporal Consistency https://github.com/phoenix104104/fast_blind_video_consistency and Universal Style Transfer https://github.com/sunshineatnoon/PytorchWCT. This proyect is the final work from EL7008 Advanced Image Processing from DIE UCH.

For the sake of clarity, I am not the owner of some programs in this repository, being the original authors listed in the previous urls. The work was to select the main functional parts of both algorithms and join them properly in order to use the webcam.

## Requirements
1. gcc 6
2. g++ 6
3. [PyTorch 0.4.0](https://pytorch.org/get-started/previous-versions/)
4. OpenCV 3.4.4
5. [Cuda 9.1](https://developer.nvidia.com/cuda-91-download-archive)
6. [TensorboardX](https://github.com/lanpa/tensorboardX)
7. Python 3.6

## Installation

The following instructions were tested with Ubuntu 18.04.

Install gcc and g++:
```
sudo apt-get install gcc-6 g++-6
``` 
Delete gcc link to executable and make a new one:
```
cd /usr/bin
sudo rm gcc
sudo ln -s gcc-6 gcc
``` 
Download Cuda 9.1 (Base installer, patch 1, patch 2 and patch 3). Follow the instructions in https://www.pugetsystems.com/labs/hpc/How-to-install-CUDA-9-2-on-Ubuntu-18-04-1184/ to install Cuda 9.1 in your system.

Install PyTorch, OpenCV and TensorboardX (highly recommended under a [virtual environment](https://virtualenv.pypa.io/en/latest/))
```
pip install https://download.pytorch.org/whl/cu91/torch-0.4.0-cp36-cp36m-linux_x86_64.whl
pip install opencv-python
pip install tensorboardX
```

## Use

## Results 
