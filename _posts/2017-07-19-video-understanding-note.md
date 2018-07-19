---
layout: post
title: "video understanding note"
categories: CNN
---

### DatSet

- UCF101: 101 classes, 13320 clips， [site](http://crcv.ucf.edu/data/UCF101.php)
- Kinetics-600: 600 classes, 392623 train, 30001 val, 72925 test, [download file](https://github.com/activitynet/ActivityNet/blob/master/Crawler/Kinetics/download.py)

### C3D

2015 年 [Learning Spatiotemporal Features with 3D Convolutional Networks](https://arxiv.org/abs/1412.0767)

on UCF101

Conv3D:  d x 3 x 3, 当 d = 3 时, 即 3 x 3 x 3, 准确率最高

类似于 VGG16, Conv 层可作为 video feature embedding


### Pseudo-3D Convolution

2017 年 [Learning Spatio-Temporal Representation With Pseudo-3D Residual Networks](http://openaccess.thecvf.com/content_iccv_2017/html/Qiu_Learning_Spatio-Temporal_Representation_ICCV_2017_paper.html) 

pytorch code: https://github.com/qijiezhao/pseudo-3d-pytorch


### I3D

Two-Stream Inflated 3D ConvNet

2018 年 [Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/abs/1705.07750)

### Non-local Neural Networks

[Non-local Neural Networks](https://arxiv.org/abs/1711.07971)

a non-local operation computes the response at a position as as weighted sum of the features at all positions in the input feature maps. The set of positions can be in space, time, spacetime, applicable for image, sequence and video problems.
   
  发现 kaiming 的文章对于 multi-label classification 问题很喜欢用 sigmoid per category 而不是 softmax，在 Mask R-CNN 中对每一个 class 设置一个 sigmoid，精度会高一些。
  
  参考 [code](https://github.com/AlexHex7/Non-local_pytorch/blob/master/lib/non_local_simple_version.py)
