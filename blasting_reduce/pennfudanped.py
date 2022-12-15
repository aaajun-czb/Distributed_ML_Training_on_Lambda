import os
import numpy as np
import torch
from PIL import Image

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import transforms as T

class PennFudanDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

'''由于PennFudan数据集很小，所以使用微调模型。此外，我们需要计算实例分割mask，所以需要使用Mask R-CNN'''
def get_model_instance_segmentation(num_classes, num_hidden=256):
    """
    获取实例分割模型
    :param num_classes:      数据集类别数
    :param num_hidden:       隐藏层节点数
    """
    # 载入在COCO数据集上的预训练模型
    # model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    # pthfile = '/var/task/resnet50_fpn.pth'
    # model_data = torch.load(pthfile)
    # model.load_state_dict(model_data)
    # # 获取模型的输入特征数
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # # 添加输出层
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # # 获取蒙版的特征数
    # in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    # # 最终模型
    # model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, num_hidden, num_classes)

    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # num_feature = model.roi_heads.box_predictor.cls_score.in_features
    # model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(num_feature, num_classes)

    # 载入想要用于替换原有模型的模型，即模型骨架
    model_bone = torchvision.models.mobilenet_v2(pretrained=True).features
    # Faster R-CNN需要知道替换模型的输出通道数
    model_bone.out_channels = 1280
    # 让RPN每个空间位置产生5 x 3锚，即需要5种不同的尺寸和3个不同的纵横比
    anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                                                        aspect_ratios=((0.5, 1.0, 2.0),))
    # 定义用于执行感兴趣区域的特征映射，以及重新定义裁剪大小
    roi_pool = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                  output_size=7,
                                                  sampling_ratio=2)
    # 重新定义模型
    model = torchvision.models.detection.FasterRCNN(model_bone, num_classes=num_classes,
                                 rpn_anchor_generator=anchor_generator,box_roi_pool=roi_pool)

    return model

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)