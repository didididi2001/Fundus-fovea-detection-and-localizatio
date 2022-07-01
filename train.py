import torch
import torch.optim as optim
import os
import numpy as np
from datasets.DetectionDataset import Detection_Dataset
from src.train_model import train_one_epoch_RetinaNet, evaluate
from src import utils
import src.transforms as T
from nets import models

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":

    torch.manual_seed(41)

    if not os.path.exists("models"):
        os.makedirs("models")

    print("Using device : %s" % device)
    device = torch.device(device)
    num_epochs = 15
    print("Num_epochs : ", num_epochs)
    print_freq = 10
    num_classes = 2

    root_dir_train = './images/train/'
    csv_fovea_train = 'fovea_localization_train_GT.csv'

    # validation data Paths
    root_dir_test = './images/val/'
    csv_fovea_test = 'fovea_localization_test_GT.csv'

    # train and test set
    train_set = Detection_Dataset(csv_fovea_train, root_dir_train, T.get_transform(train=True),
                                  box_width_Fovea=(120, 120), image_size=(800, 800))

    test_set = Detection_Dataset(csv_fovea_test, root_dir_test, T.get_transform(train=False),
                                 box_width_Fovea=(120, 120), image_size=(800, 800))

    data_loader = torch.utils.data.DataLoader(
        train_set, batch_size=1, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    test_data_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    model = models.resnet101(num_classes=2, pretrained=True)
    model.to(device)
    model.training = True
    model.train()
    model.freeze_bn()
    train_one_epoch = train_one_epoch_RetinaNet
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, verbose=True)
    max_iou = 0.0
    for epoch in range(num_epochs):
        epoch_loss = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq)
        iou = evaluate(model, test_set, device)
        scheduler.step(np.mean(epoch_loss))
        # 保存最大iou的模型
        if iou > max_iou:
            max_iou = iou
            save_path = "./models/model.pth"
            print("Saving checkpoint {:s} at epoch {:d}".format(save_path, epoch))
            torch.save(model.state_dict(), save_path)

