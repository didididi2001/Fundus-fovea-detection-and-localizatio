import matplotlib.pyplot as plt
from src.utils import get_center

def plot_prediction(img, Fovea_true_box, Fovea_predicted_box, idx):
    # Get bounding box coordinates
    _xx1, _yy1, _xx2, _yy2 = Fovea_true_box
    Fovea_true_center = get_center(Fovea_true_box)
    # return [(x1 + x2) / 2, (y1 + y2) / 2]

    # Retrieve predicted bounding boxes and scores
    x1, y1, x2, y2 = Fovea_predicted_box
    Fovea_predicted_center = get_center(Fovea_predicted_box)
    # return [(x1 + x2) / 2, (y1 + y2) / 2]
    plt.figure(figsize=(10, 10))
    # 本来是通道，长宽高
    plt.imshow(img.mul(255).permute(1, 2, 0).byte().numpy(), cmap="gray")
    # 绘制真实框
    plt.plot([_xx1, _xx1, _xx2, _xx2, _xx1], [_yy1, _yy2, _yy2, _yy1, _yy1], 'b-', label="ground truth Fovea")
    # 绘制预测框
    plt.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], 'w-', label="predicted Fovea")
    # 绘制真实中心点
    plt.scatter(Fovea_true_center[0], Fovea_true_center[1], c='b', marker='o')
    # 绘制预测中心点
    plt.scatter(Fovea_predicted_center[0], Fovea_predicted_center[1], c='w', marker='o')
    plt.legend()
    plt.savefig('./figures/{}.png'.format(idx))
    plt.axis('off')
    # plt.show()
