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

    # 训练集路径
    root_dir_train = './images/train/'
    csv_fovea_train = './fovea_localization_train_GT.csv'

    # 测试路径
    root_dir_test = "./images/val/"
    csv_fovea_test = './fovea_localization_test_GT.csv'
    # ture_out = pd.read_csv(csv_fovea_test)
    # ture_out = np.array([ture_out.iloc[:, 1], ture_out.iloc[:, 2]]).astype('float')
    # ture_out = ture_out.transpose()
    # 训练和测试数据加载
    # train_set = Detection_Dataset(csv_fovea_train, root_dir_train, T.get_transform(train=True),
    #                               box_width_Fovea=(120, 120), image_size=(800, 800))
    test_set = Detection_Dataset(csv_fovea_test, root_dir_test, T.get_transform(train=False),
                                 box_width_Fovea=(120, 120), image_size=(800, 800))

    model = models.resnet101(num_classes=2, pretrained=True)

    model.load_state_dict(model_state_dict)

    model.to(device)
    dataset = test_set
    text_out = []
    if not os.path.exists("./figures"):
        os.makedirs("./figures")
    for i in range(16):

        img, Fovea_true_box, Fovea_predicted_box = utils.get_boxes(model, dataset, threshold=0.08, img_idx=i)
        plot_prediction(img, Fovea_true_box, Fovea_predicted_box, i)
        _, target, scale_factor = dataset[i]
        # 生成预测中心坐标
        predicted_center = get_center(Fovea_predicted_box)
        predicted_center.insert(0, i)
        predicted_center[1] = predicted_center[1] / scale_factor[0]
        predicted_center[2] = predicted_center[2] / scale_factor[1]
        text_out.append(predicted_center)


    # 计算预测中心坐标与实际中心坐标平均距离
    final_out = np.array(text_out)
    # loss = (final_out[:, 1]-ture_out[:, 0])**2+(final_out[:, 2]-ture_out[:, 1])**2
    # loss = np.sqrt(loss).sum() / ture_out.shape[0]
    # print("average distance:", loss)
    #  预测坐标储存
    file = open(root_dir_test+"pre.txt", "w")
    np.savetxt(file, final_out, fmt=["%d", "%.6f", "%.6f"], delimiter=',')
    file.close()


