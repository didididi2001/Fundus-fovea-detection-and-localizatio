from PIL import Image
import torch
import os
import pandas as pd
import numpy as np
from datasets.DetectionDataset import Detection_Dataset
from src import utils
import src.transforms as T
from nets import models
from src.plots import plot_prediction
from src.utils import get_center
from pre_deal import augment, reshape ,test_get_boxes
device = "cuda" if torch.cuda.is_available() else "cpu"
if __name__ == "__main__":
    # 为CPU中设置种子，生成随机数：
    # 在需要生成随机数据的实验中，每次实验都需要生成数据。
    # 设置随机种子是为了确保每次生成固定的随机数，
    # 这就使得每次实验结果显示一致了，有利于实验的比较和改进。
    torch.manual_seed(41)

    if not os.path.exists("models"):
        os.makedirs("models")

    print("Using device : %s" % device)
    device = torch.device(device)
    num_classes = 2

    CHECKPOINT_FILE = "./models/model.pth"
    print("Checkpoint file: {:s}".format(CHECKPOINT_FILE))
    model_state_dict = torch.load(CHECKPOINT_FILE)
    model = models.resnet101(num_classes=2, pretrained=True)

    model.load_state_dict(model_state_dict)
    model.to(device)
    # 测试图片相较于本检测文件路径
    # test_dir_path = "../../test"
    test_dir_path = "./images/val/"
    #测试图片名列表
    img_name_list = os.listdir(test_dir_path)
    text_out = []
    # 逐个读入测试数据
    idx = 0
    for img_name in img_name_list:
        # 读入测试图片
        image = Image.open(os.path.join(test_dir_path, img_name))
        image = augment(image)
        image, scale_factor = reshape(image)
        boxes = [0,0,0,0]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # create labels
        labels = torch.tensor([1], dtype=torch.int64)
        image_id = torch.tensor([idx])
        iscrowd = torch.zeros((1), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = torch.tensor([0], dtype=torch.int64)
        target["iscrowd"] = iscrowd
        img, Fovea_predicted_box = test_get_boxes(model, image, target, threshold=0.08, img_idx=idx)
        predicted_center = get_center(Fovea_predicted_box)
        data = img_name.split(".")[0]
        print(type(data))
        predicted_center.insert(0, data)
        predicted_center[1] = predicted_center[1] / scale_factor[0]
        predicted_center[2] = predicted_center[2] / scale_factor[1]
        # print(predicted_center)
        text_out.append(predicted_center)
        idx = idx + 1
np.savetxt("pre.txt", np.array(text_out), fmt='%s',delimiter=',')




