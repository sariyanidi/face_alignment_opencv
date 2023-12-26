# Introduction

This repository contains a minimal-dependency facial alignment method. We re-implement a state-of-the-art landmark detection method (2D-FAN) using only OpenCV. Thus, we refer to the software in this repository as FAO (Facial Aligment with OpenCV).

## Installation
The simplest way to install FAO is to use `pip` as follows. **But**, this will be slow (~7-8 FPS) because the default `opencv-python` package has no CUDA support, therefore FAO will run on CPU. (See more below for GPU processing)
```
git clone https://github.com/sariyanidi/face_alignment_opencv.git
cd face_alignment_opencv
pip install --upgrade pip
pip install -r requirements_cpu.txt 
```

If you want to use FAO with GPU support (~100 FPS), you need to manually compile OpenCV with CUDA and CUDNN support. In this case, you should make sure that the repository does *not* use the OpenCV installed through `pip`, but the one that you manually compiled. Once you manually install OpenCV, then the following lines will suffice to use FAO with cuda support.
```
git clone https://github.com/sariyanidi/face_alignment_opencv.git
cd face_alignment_opencv
pip install -r requirements_gpu.txt 
```


## Testing

FAO comes with three scripts for testing. The first one is for single-image processing, which can be tested as
```
python3 process_image.py samples/corb.jpeg
python3 process_image.py samples/barca.jpg
```
The second script is for video processing, which can be tested as:
```
python3 process_video.py samples/la_boheme.mp4
```
The third script is for processing all images in a directory, which can be tested as:
```
python3 process_image_dir.py samples/testdir
```
