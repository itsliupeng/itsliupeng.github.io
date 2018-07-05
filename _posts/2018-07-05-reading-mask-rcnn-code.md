---
layout: post
title:  "Mask R-CNN 源码解读"
categories: CNN
---

跟踪一下 [Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch) Mask R-CNN train 和 inference 的流程

Mask R-CNN 是在 Faster R-CNN 基础上把 RoIPool 替换为 RoIAlign，即使用 bilinear interpolation （双线性差值）方式把  feature map 缩小为固定大小（如 7 x 7 ），所以理解 Faster R-CNN 是基础。

除了论文，在网上找到的最详细的文章是 [Object Detection and Classification using R-CNNs](http://www.telesens.co/2018/03/11/object-detection-and-classification-using-r-cnns/)

