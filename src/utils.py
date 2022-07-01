from __future__ import print_function
from src.metrics import non_max_suppression
import torch

import numpy as np


def get_annotations_retinanet(target):
    """获取相应的批注"""
    annotations = np.zeros((0, 5))
    annotation = np.zeros((1, 5))
    # boxes的四个坐标
    annotation[0, 0] = target["boxes"][0][0]
    annotation[0, 1] = target["boxes"][0][1]
    annotation[0, 2] = target["boxes"][0][2]
    annotation[0, 3] = target["boxes"][0][3]
    # 类别
    annotation[0, 4] = 1
    # 维度扩充
    annotations = np.append(annotations, annotation, axis=0)
    annotations = np.expand_dims(annotations, axis=0)
    annotations = torch.from_numpy(annotations).float()

    return annotations


def get_boxes(model, dataset, threshold=0.008, img_idx=0):
    # 预测框
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    img, target, _ = dataset[img_idx]
    model.eval()
    # 得到真实框
    Fovea_true_box = target["boxes"][0]

    with torch.no_grad():
        # unsqueeze主要起到升维的作用，后续图像处理可以更好地进行批操作
        scores, labels, boxes = model(img.unsqueeze(0).cuda())
        # 框的得分 ，标签，以及位置
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        boxes = boxes.cpu().numpy()


    labels = labels[0:]
    scores = scores[0:]
    Fovea_boxes = boxes[0:]

    # 进行极大值抑制得到最终的锚框
    if len(Fovea_boxes) > 0:
        kept_idx = list(non_max_suppression(Fovea_boxes, scores, threshold))
        Fovea_boxes = [list(boxes[0:][i]) for i in range(len(boxes[0:]))]
        if len(kept_idx) == 0:
            Fovea_predicted_box = boxes[0]
        else:
            Fovea_predicted_box = Fovea_boxes[kept_idx[0]]
    else:
        print("Fovea boxes empty for img ", img_idx)
        Fovea_predicted_box = boxes[0]

    return img, Fovea_true_box, Fovea_predicted_box


def get_center(box):
    # 计算框的中心坐标
    x1, y1, x2, y2 = box
    return [(x1 + x2) / 2, (y1 + y2) / 2]


def get_center_distance(boxA, boxB, factor=None):
    # 计算框中心的距离
    centA = get_center(boxA)
    centB = get_center(boxB)
    return np.sqrt(
        ((1 / factor[0]) ** 2) * (centA[0] - centB[0]) ** 2 + ((1 / factor[1]) ** 2) * (centA[1] - centB[1]) ** 2)

def collate_fn(batch):
    return tuple(zip(*batch))







