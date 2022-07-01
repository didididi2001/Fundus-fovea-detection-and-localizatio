import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from albumentations import RandomBrightnessContrast, RandomGamma, CLAHE, Compose
import warnings

warnings.filterwarnings("ignore")


class Detection_Dataset(Dataset):
    # 对Dataset进行改写
    def __init__(self, csv_fovea, root_dir, transform=None, box_width_Fovea=(120, 120), image_size=(800, 800)):

        self.fovea = pd.read_csv(csv_fovea)
        self.root_dir = root_dir
        self.transform = transform
        self.box_width_Fovea = box_width_Fovea
        self.image_size = image_size

    def __len__(self):

        for i, id in enumerate(self.fovea['data']):
            if not isinstance(id, int):
                break
        return i

    def __augment__(self, im):
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


    def __reshape__(self, sample):

        # resize + to tensor
        scale_factor = np.ones(2)
        image = sample['image']
        init_shape = np.array(list(image.size))
        scale = transforms.Resize(self.image_size)
        # transforms.Resize([224, 224])就能将输入图片转化成224×224的输入特征图
        # 虽然会改变图片的长宽比，但是本身并没有发生裁切，仍可以通过resize方法返回原来的形状
        to_tensor = transforms.ToTensor()
        # 把一个取值范围是[0, 255]的PIL.Image或者shape为(H, W, C)的numpy.ndarray，
        # 转换成形状为[C, H, W]，
        # 取值范围是[0, 1.0]的torch.FloadTensor
        composed = transforms.Compose([scale, to_tensor])
        sample['image'] = composed(sample['image'])
        final_shape = np.array([sample['image'].shape[1], sample['image'].shape[2]])
        # update coordinates
        if not set(final_shape) == set(init_shape):
            scale_factor *= (final_shape / init_shape)
            sample['Fovea'] *= scale_factor
        # print("sample, scale_factor",sample, scale_factor)
        return sample, scale_factor

    def __get_boxes__(self, sample):

        width = self.box_width_Fovea
        bbox = []
        bbox.append(sample['Fovea'][0] - width[0] / 2)
        bbox.append(sample['Fovea'][1] - width[1] / 2)
        bbox.append(sample['Fovea'][0] + width[0] / 2)
        bbox.append(sample['Fovea'][1] + width[1] / 2)
        # print("bbox",bbox)
        return bbox

    def __getitem__(self, idx):

        # format index
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get coordinates
        Fovea = np.array([self.fovea.iloc[idx, 1], self.fovea.iloc[idx, 2]]).astype('float')
        img_name = str(self.fovea.iloc[idx, 0])

        # read image
        img_path = os.path.join(self.root_dir, img_name + '.jpg')
        image = Image.open(img_path)
        image = self.__augment__(image)

        # create the sample dictionary
        sample = {'image': image, 'Fovea': Fovea}
        # reshape the image and update coordinates
        sample, scale_factor = self.__reshape__(sample)

        # create bounding boxes
        boxes = []
        boxes.append(self.__get_boxes__(sample))
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # create labels
        labels = torch.tensor([1], dtype=torch.int64)

        # image_id
        image_id = torch.tensor([idx])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((1), dtype=torch.int64)

        # create target dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = torch.tensor([self.box_width_Fovea[0] * self.box_width_Fovea[1]], dtype=torch.int64)
        target["iscrowd"] = iscrowd

        # apply transformations
        img = sample['image']
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target, scale_factor
