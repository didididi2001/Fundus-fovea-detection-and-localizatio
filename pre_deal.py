import torch
import pandas as pd
from torchvision import transforms
from PIL import Image
import numpy as np
from albumentations import RandomBrightnessContrast, RandomGamma, CLAHE, Compose
from src.metrics import non_max_suppression
def augment( im):
    image = np.array(im)
    # 变换范围到0-255
    image = np.uint8(image)
    # 用cv2.normalize函数配合cv2.NORM_MINMAX，可以设置目标数组的最大值和最小值，
    # 然后让原数组等比例的放大或缩小到目标数组，如下面的例子中是将img的所有数字等比例的放大或缩小到0–255
    # 范围的数组中，
    # cv2.normalize(img, out, 0, 255, cv2.NORM_MINMAX)
    # 然后改变数据类型
    # np.array([out], dtype=‘uint8’)
    light = Compose([
        # RandomBrightnessContrast(p=1),
        # RandomGamma(p=1),
        CLAHE(p=1),
    ], p=1)
    image = light(image=image)['image']
    # 这是一个图像增强的pipline, 其中的流程是：
    #
    # Resize就是拉伸图片修改尺寸
    # RandomGamma就是使用gamma变换
    # RandomBrightnessContrast就是随机选择图片的对比度和亮度
    # CLAHE是一种对比度受限情况下的自适应直方图均衡化算法
    # HorizontalFlip水平翻转
    # ShiftScaleRotate这个就是平移缩放旋转三合一，给力！
    # Normalize这个就是图像归一化了。
    return Image.fromarray(image)


def reshape(image):
    # resize + to tensor
    image_size = (800, 800)
    scale_factor = np.ones(2)
    init_shape = np.array(list(image.size))
    scale = transforms.Resize(image_size)
    # transforms.Resize([224, 224])就能将输入图片转化成224×224的输入特征图
    # 虽然会改变图片的长宽比，但是本身并没有发生裁切，仍可以通过resize方法返回原来的形状
    to_tensor = transforms.ToTensor()
    # 把一个取值范围是[0, 255]的PIL.Image或者shape为(H, W, C)的numpy.ndarray，
    # 转换成形状为[C, H, W]，
    # 取值范围是[0, 1.0]的torch.FloadTensor
    composed = transforms.Compose([scale, to_tensor])
    image = composed(image)
    final_shape = np.array([image.shape[1], image.shape[2]])
    # update coordinates
    if not set(final_shape) == set(init_shape):
        scale_factor *= (final_shape / init_shape)
    # print("sample, scale_factor",sample, scale_factor)
    return image, scale_factor
def test_get_boxes(model, img,target, threshold=0.008, img_idx=0):
    # 预测框
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
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

    return img, Fovea_predicted_box