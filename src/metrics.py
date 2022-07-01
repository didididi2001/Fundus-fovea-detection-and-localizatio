import numpy as np


def compute_iou(box, boxes, box_area, boxes_area):
    # 计算相交的面积
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    # 将俩个boxex的面积相加减去相交的面积
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def non_max_suppression(boxes, scores, threshold):

    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)

    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    area = (y2 - y1) * (x2 - x1)
    # 得分从大到小排序
    ixs = scores.argsort()[::-1]
    pick = []
    while len(ixs) > 0:
        # 找到最高得分的，加入结果
        i = ixs[0]
        pick.append(i)
        # 计算其与其他候选框的IOU
        iou = compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
        # 找到IOU大于threshold的，剔除掉
        remove_ixs = np.where(iou > threshold)[0] + 1
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


def accuracy(gt_labels, pred_labels):
    return len(np.where(gt_labels == pred_labels)[0]) / len(gt_labels)


def f1_score(gt_labels, pred_labels):
    tp = len(gt_labels == pred_labels)
    fp = len(np.where(pred_labels[np.where(gt_labels == 2)] == 1)[0])
    fn = len(np.where(pred_labels[np.where(gt_labels == 1)] == 2)[0])
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print('Precision: {:.3f}'.format(precision))
    print('Recall: {:.3f}'.format(recall))
    return 2 * precision * recall / (precision + recall)


def IoU(boxA, boxB):
    # 计算IOU
    xA1, yA1, xA2, yA2 = boxA
    xB1, yB1, xB2, yB2 = boxB
    x1 = max(xA1, xB1)
    y1 = max(yA1, yB1)
    x2 = min(xA2, xB2)
    y2 = min(yA2, yB2)
    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    boxAArea = (xA2 - xA1 + 1) * (yA2 - yA1 + 1)
    boxBArea = (xB2 - xB1 + 1) * (yB2 - yB1 + 1)
    union = boxAArea + boxBArea - intersection
    return intersection / union