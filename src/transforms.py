import random
from PIL import ImageEnhance
from torchvision.transforms import functional as F
from torchvision import transforms

class Compose(object):
    def __init__(self, transformations):
        self.transformations = transformations

    def __call__(self, image, target):
        for t in self.transformations:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        # 概率
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            # 水平翻转
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return image, target


class ImageEnhencer:
    def __call__(self, image, target):
        image = transforms.ToPILImage()(image).convert("RGB")

        # 亮度
        enh_bri = ImageEnhance.Brightness(image)
        brightness = round(random.uniform(0.8, 1.2), 2)
        image = enh_bri.enhance(brightness)

        # 颜色
        enh_col = ImageEnhance.Color(image)
        color = round(random.uniform(0.8, 1.2), 2)
        image = enh_col.enhance(color)

        # 对比度
        enh_con = ImageEnhance.Contrast(image)
        contrast = round(random.uniform(0.8, 1.2), 2)
        image = enh_con.enhance(contrast)
        # image = np.array(image)

        return transforms.ToTensor()(image), target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


def get_transform(train):
    transform = []

    if train:
        # 作用于训练时
        transform.append(RandomHorizontalFlip(0.2))
        transform.append(ImageEnhencer())

    return Compose(transform)
