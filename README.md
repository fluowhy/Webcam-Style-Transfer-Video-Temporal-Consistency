# Style Transfer Video Temporal Consistency for Webcam

The following proyect is a merge from Learning Blind Video Temporal Consistency https://github.com/phoenix104104/fast_blind_video_consistency and Universal Style Transfer https://github.com/sunshineatnoon/PytorchWCT. This proyect is the final work from EL7008 Advanced Image Processing from DIE UCH.

For the sake of clarity, I am not the owner of some programs in this repository, being the original authors listed in the previous urls. The work done consisted to select the functional main parts of both algorithms and join them properly in order to use the webcam.

## Table of contents
1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Use](#use)
3. [Results](#results)

## Requirements <a name="requirements"></a>

* Python 3.6
* gcc 6
* g++ 6
* [PyTorch 0.4.0](https://pytorch.org/get-started/previous-versions/)
* Torchvision
* OpenCV 3.4.4
* [Cuda 9.1](https://developer.nvidia.com/cuda-91-download-archive)
* [TensorboardX](https://github.com/lanpa/tensorboardX)
* colorama
* tqdm
* setproctitle


## Installation <a name="installations"></a>

The following instructions were tested on Ubuntu 18.04.

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

Install PyTorch, OpenCV, TensorboardX and others (highly recommended under a [virtual environment](https://virtualenv.pypa.io/en/latest/))
``` 
pip install https://download.pytorch.org/whl/cu91/torch-0.4.0-cp36-cp36m-linux_x86_64.whl
pip install opencv-python
pip install tensorboardX
pip install colorama
pip install tqdm
pip install setproctitle
```
Clone the repository:
```
git clone https://github.com/fluowhy/Webcam-Style-Transfer-Video-Temporal-Consistency.git
```
Download pretrained blind video temporal consistency model as stated [here](https://github.com/phoenix104104/fast_blind_video_consistency#learning-blind-video-temporal-consistency):
```
cd pretrained_models
./download_models.sh
cd ..
```
[Download](https://drive.google.com/file/d/1M5KBPfqrIUZqrBZf78CIxLrMUT4lD4t9/view) VGG encoder and decoder pretrained models as stated [here](https://github.com/sunshineatnoon/PytorchWCT) and extract it in the repository folder. There should be a /models folder with VGG encoder and decoder models inside it.

Please note that some Python libraries would be missing in your environment, but they must be easy to install with pip. 

## Use <a name="use"></a>

Now it is possible to run the algorithm without problems. To start it run:
```
python run_webcam.py --cuda --alpha <value> --style <path/to/style/image> 
```
where --cuda enables cuda, --alpha [0, 1] is a parameter which tunes the style component in the processed image and --style is the path to the style image. Default values are --alpha 0.5 and --style style/kandinsky.jpg, Yellow-Red-Blue painting by Kandinsky.

## Results <a name="results"></a>

![](result.png?raw=true)

The screenshot is the actual video from the webcam and processed images. The fps bottleneck seems to be the style transfer algorithm. 