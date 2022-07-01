from src import utils
import numpy as np
from tqdm import tqdm
from src.metrics import IoU
import torch


def train_one_epoch_RetinaNet(model, optimizer, data_loader, device, epoch, print_freq):
    """
    模型的正向传播、反向传播
    """
    model.train()
    epoch_loss = []

    print("\n------------------------Training---------------------------------------\n")
    for iter_num, (img, target, factor) in enumerate(data_loader):
        annotations = utils.get_annotations_retinanet(target[0])
        img = img[0].unsqueeze(0)
        # print(img.size(), annotations.size())
        optimizer.zero_grad()
        classification_loss, regression_loss = model([img.to(device), annotations.to(device)])
        try:
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            loss = classification_loss + regression_loss
            if bool(loss == 0):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            epoch_loss.append(float(loss))
            if iter_num % print_freq == 0:
                print(
                    'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch, iter_num, float(classification_loss), float(regression_loss), np.mean(epoch_loss)))
            del classification_loss
            del regression_loss
        except Exception as e:
            print(e)
            continue
    return epoch_loss


def evaluate(model, dataset, device):
    """"
    用平均IOU来评估一个模型的好坏
    """
    model.eval()
    dist_Fovea = 0
    gt_boxes_Fovea, pred_boxes_Fovea = np.empty(4), np.empty(4)
    print("\n------------------------Validation---------------------------------------\n")
    for idx in tqdm(range(dataset.__len__())):
        img, target, factor = dataset[idx]
        _, Fovea_true_box, Fovea_predicted_box = utils.get_boxes(model, dataset, threshold=0.08, img_idx=idx)
        # 将真实的锚框叠加起来
        gt_boxes_Fovea = np.vstack((gt_boxes_Fovea, Fovea_true_box))
        # 将预测的锚框叠加起来
        pred_boxes_Fovea = np.vstack((pred_boxes_Fovea, Fovea_predicted_box))
        # 计算真实锚框与预测锚框之间的距离
        dist_Fovea += utils.get_center_distance(Fovea_true_box, Fovea_predicted_box, factor)

    # 计算平均IOU
    average_iou_Fovea = np.mean([IoU(gt_boxes_Fovea[i], pred_boxes_Fovea[i]) for i in range(len(gt_boxes_Fovea))])
    print("Average IoU for Fovea detection over validation set: {:.3f}".format(average_iou_Fovea))
    # 计算锚框中心点的距离
    mean_dist_Fov = dist_Fovea / dataset.__len__()
    print('Mean distance between Fovea centers over validation set: {:.3f} \n'.format(mean_dist_Fov))
    # 返回的时预测锚框与真实锚框之间的平均IOU
    # 其实也可以时预测锚框与真实锚框之间的平均距离
    return average_iou_Fovea

