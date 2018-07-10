---
layout: post
title:  "Mask R-CNN 源码解读"
categories: CNN
---

跟踪一下 [Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch) Mask R-CNN train 和 inference ¡的流程

Mask R-CNN 是在 Faster R-CNN 基础上把 RoIPool 替换为 RoIAlign，即使用 bilinear interpolation （双线性差值）方式把  feature map 缩小为固定大小（如 7 x 7 ），所以理解 Faster R-CNN 是基础。

Faster R-CNN, 除了论文，在网上找到的最详细的文章是 [Object Detection and Classification using R-CNNs](http://www.telesens.co/2018/03/11/object-detection-and-classification-using-r-cnns/)


![Faster R-CNN training](/images/2018-07-05-reading-mask-rcnn-code/faster_rcnn.jpg)

分为 RPN 网络和 R-CNN 网络。RPN 输出可用的 proposals, 其中 classification 用来分类 anchor box ，可以分为 foreground（比如 > 0.5）和 background（比如 < 0.5）, regression 将 anchor box 回归修正成 roi proposals，然后在 ROIPool(或 ROIAlign) 层，extract 成固定大小的 feature map（比如 7 x 7）

### inference
整个 inference 的代码封装的比较好

```python
# load config
dataset = datasets.get_coco_dataset()
cfg.MODEL.NUM_CLASSES = len(dataset.classes)
cfg_from_file('configs/baselines/e2e_mask_rcnn_R-50-C4_1x.yaml')
cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = False  # Don't need to load imagenet pretrained weights
assert_and_infer_cfg()

# load model
maskRCNN = Generalized_RCNN()
checkpoint = torch.load('/home/work/liupeng11/code/Detectron.pytorch/models/e2e_mask_rcnn_R-50-C4_1x.pth', map_location=lambda storage, loc: storage)
net_utils.load_ckpt(maskRCNN, checkpoint['model'])
maskRCNN.eval()
maskRCNN = mynn.DataParallel(maskRCNN, cpu_keywords=['im_info', 'roidb'], minibatch=True, device_ids=[0])

# load image
img_path = "/home/work/liupeng11/code/Detectron.pytorch/demo/sample_images/img1.jpg"
im = cv2.imread(img_path)

# detect bouding boxes and segments
from core.test import im_detect_bbox, im_detect_mask, box_results_with_nms_and_limit, segm_results
scores, boxes, im_scale, blob_conv = im_detect_bbox(maskRCNN, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, None)
scores, boxes, cls_boxes = box_results_with_nms_and_limit(scores, boxes)
masks = im_detect_mask(maskRCNN, im_scale, boxes, blob_conv)
cls_segms = segm_results(cls_boxes, masks, boxes, im.shape[0], im.shape[1])
cls_keyps = None

# save detected image
name = 'test'
output_dir = '/home/work/liupeng11/code/Detectron.pytorch/tmp'
vis_utils.vis_one_image(
    im[:, :, ::-1],  # BGR -> RGB for visualization
    name,
    output_dir,
    cls_boxes,
    cls_segms,
    cls_keyps,
    dataset=dataset,
    box_alpha=0.3,
    show_class=True,
    thresh=0.7,
    kp_thresh=2,
    ext='jpg',
)
```

关键代码在 `im_detect_bbox` 中，这一部分是 Faster-RCNN 检测 bouding boxes, mask 分割在 `im_detect_mask` 中

### im_detect_bbox

在 inference 时， maskRCNN 的 forward 方法返回 cls_score， bbox_pred 以及 RPN 网络的 rois，不涉及 mask 网络，所以此时的 maskRCNN 是一个 Faster R-CNN 网络

```python
return_dict['rois'] = rpn_ret['rois']
return_dict['cls_score'] = cls_score
return_dict['bbox_pred'] = bbox_pred
```
maskRCNN 的 forward 可以简化成如下过程：

```python
# prepare input and model
pil = Image.open(img_path).convert('RGB')
trans = transforms.Compose([
    transforms.Resize((800, 600)),
    transforms.ToTensor()
    ])

x = Variable(torch.unsqueeze(trans(pil), 0))
x = x.cuda()
m = maskRCNN.module
m.eval()

# feature_map
blob_conv = m.Conv_Body(x)

# RPN network
rpn_conv = F.relu(m.RPN.RPN_conv(blob_conv), inplace=False)
rpn_cls_logits = m.RPN.RPN_cls_score(rpn_conv)
rpn_bbox_pred = m.RPN.RPN_bbox_pred(rpn_conv)
rpn_cls_prob = F.sigmoid(rpn_cls_logits)

# genrete proposals (rois)
im_info = Variable(torch.Tensor([[800, 600, 1]]))
rpn_rois, rpn_rois_prob = m.RPN.RPN_GenerateProposals(rpn_cls_prob, rpn_bbox_pred, im_info)

rpn_ret = {'rpn_cls_logits': rpn_cls_logits, 'rpn_bbox_pred': rpn_bbox_pred, 'rpn_rois': rpn_rois, 'rpn_roi_probs': rpn_rois_prob}
rpn_ret['rois'] = rpn_ret['rpn_rois']

# bouding box network
box_feat = m.Box_Head(blob_conv, rpn_ret)
cls_score, bbox_pred = m.Box_Outs(box_feat)

```

