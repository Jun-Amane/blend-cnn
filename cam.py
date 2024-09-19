import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.transforms import transforms
from torchvision import models
from torchsummary import summary
from matplotlib import pyplot as plt
import numpy as np
import cv2

from ALKA import ALKA
from dataset import AlkaDataset

from util import *


def myimshows(imgs, titles=False, fname="test.jpg", size=6):
    lens = len(imgs)
    fig = plt.figure(figsize=(size * lens, size))
    if titles == False:
        titles = "0123456789"
    for i in range(1, lens + 1):
        cols = 100 + lens * 10 + i
        plt.xticks(())
        plt.yticks(())
        plt.subplot(cols)
        if len(imgs[i - 1].shape) == 2:
            plt.imshow(imgs[i - 1], cmap='Reds')
        else:
            plt.imshow(imgs[i - 1])
        plt.title(titles[i - 1])
    plt.xticks(())
    plt.yticks(())
    plt.savefig(fname, bbox_inches='tight')
    plt.show()


def tensor2img(tensor, heatmap=False, shape=(300, 300)):
    np_arr = tensor.detach().numpy()[0]
    # 对数据进行归一化
    if np_arr.max() > 1 or np_arr.min() < 0:
        np_arr = np_arr - np_arr.min()
        np_arr = np_arr / np_arr.max()
    np_arr = (np_arr * 255).astype(np.uint8)
    if np_arr.shape[0] == 1:
        np_arr = np.concatenate([np_arr, np_arr, np_arr], axis=0)
    np_arr = np_arr.transpose((1, 2, 0))
    if heatmap:
        np_arr = cv2.resize(np_arr, shape)
        np_arr = cv2.applyColorMap(np_arr, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    return np_arr / 255


def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())
    print("backward_hook:", grad_in[0].shape, grad_out[0].shape)


def farward_hook(module, input, output):
    fmap_block.append(output)
    print("farward_hook:", input[0].shape, output.shape)


# 加载模型
model = torch.load('best.pt', map_location=torch.device('cpu'))
model.eval()  # 评估模式
# summary(model,input_size=(3,512,512))

# 注册hook
fh = model.image_model.avgpool.register_forward_hook(farward_hook)
bh = model.image_model.avgpool.register_backward_hook(backward_hook)

# 定义存储特征和梯度的数组
fmap_block = list()
grad_block = list()

# 加载变量并进行预测
path = r"../../dataset/wheat_img/Wheat Red Spider/WheatRedSpider_0.jpg"
bin_data = torchvision.io.read_file(path)  # 加载二进制数据
img = torchvision.io.decode_image(bin_data) / 255  # 解码成CHW的图片
img = img.unsqueeze(0)  # 变成BCHW的数据，B==1; squeeze
img = torchvision.transforms.functional.resize(img, [300, 300])
preds = model(img, torch.tensor([0]))
print("pred type:", preds.argmax(1))

# 构造label，并进行反向传播
clas = 0  #
trues = torch.ones((1,), dtype=torch.int64) * clas
ce_loss = nn.CrossEntropyLoss()
loss = ce_loss(preds, trues)
loss.backward()

# 卸载hook
fh.remove()
bh.remove()

# 取出相应的特征和梯度
layer1_grad = grad_block[-1]  # layer1_grad.shape [1, 64, 128, 128]
layer1_fmap = fmap_block[-1]

# 将梯度与fmap相乘
cam = layer1_grad[0, 0].mul(layer1_fmap[0, 0])
for i in range(1, layer1_grad.shape[1]):
    cam += layer1_grad[0, i].mul(layer1_fmap[0, i])
layer1_grad = layer1_grad.sum(1, keepdim=True)  # layer1_grad.shape [1, 1, 128, 128]
layer1_fmap = layer1_fmap.sum(1, keepdim=True)  # 为了统一在tensor2img函数中调用
cam = cam.reshape((1, 1, *cam.shape))

# 进行可视化
img_np = tensor2img(img, shape=(300, 300))
# layer1_fmap=torchvision.transforms.functional.resize(layer1_fmap,[224, 224])
layer1_grad_np = tensor2img(layer1_grad, heatmap=True, shape=(300, 300))
layer1_fmap_np = tensor2img(layer1_fmap, heatmap=True, shape=(300, 300))
cam_np = tensor2img(cam, heatmap=True, shape=(300, 300))
print("颜色越深（红），表示该区域的值越大")
# myimshows([img_np, cam_np, cam_np * 0.4 + img_np * 0.6], ['image', 'cam', 'cam + image'])
plt.axis('off')
plt.imshow((cam_np * 0.4 + img_np * 0.6), cmap='Reds')
plt.savefig('test.jpg')

