"""
---
title: U-Net
summary: >
    PyTorch implementation and tutorial of U-Net model.
---

# U-Net

This is an implementation of the U-Net model from the paper,
[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597).

U-Net consists of a contracting path and an expansive path.
The contracting path is a series of convolutional layers and pooling layers,
where the resolution of the feature map gets progressively red
Expansive path is a series of up-sampling layers and convolutional layers
where the resolution of the feature map gets progressively increased.

At every step in the expansive path the corresponding feature map from the contracting path
concatenated with the current feature map.

![U-Net diagram from paper](unet.png)

Here is the [training code](experiment.html) for an experiment that trains a U-Net
on [Carvana dataset](carvana.html).
"""
import csv
import math
import cmath
import os


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from scipy.fftpack import fft, ifft, fft2,ifft2,ifftshift
from scipy import stats
# 定义超参数
batch_size=10  # 每次处理的数据数量
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")     # 用GPU或CPU训练
epoch=300    # 训练数据集训练轮次

import torch
import csv
from torch.utils.data import Dataset


class MyDataset(Dataset):  # 继承Datasets
    def __init__(self, dir_data):  # 初始化一些用到的参数，一般不仅有self
        # 读取散射场数据
        with open(os.path.join(dir_data, 'magnitude_Y_results_1_2.csv'), 'r') as f:
            reader = csv.reader(f)
            temp = []  # 存储散射场数据
            for row in reader:
                data_temp = list(map(float, row))  # 数据转为float型
                temp.append(data_temp)

        # 读取标签数据
        with open(os.path.join(dir_data, 'phase_Y_results_1.csv'), 'r') as l:
            reader = csv.reader(l)
            label = []  # 存储标签
            for row in reader:
                label_temp = list(map(float, row))
                label.append(label_temp)

        angle = 240  # 入射角个数
        op_num = 240  # 观测点个数
        m = 0
        n = 0
        res = []  # 元素为元组形式（data，label），data为shape为[2,8,160]的tensor,label为shape为[160]的tensor


        for i in range(1000):
            data_amp1_amp2 = []  #
            for _ in range(2):  # 每组数据两个通道 分别为近场1近场2
                data_1 = []
                for _ in range(angle):  # 每个通道angle行，即angle个角度
                    data_1.append(temp[m])
                    m += 1
                data_amp1_amp2.append(data_1)  # 存储实部和虚部

            data_amp1_amp2_normalisation = [[], []]
            data_1_phase = []  # 存储plane1的相位
            label_phase = []

            for row in range(angle):
                data_amp1_temp = []
                data_amp2_temp = []
                label_phase.append(label[n])
                for column in range(op_num):
                    data_amp1_temp.append(data_amp1_amp2[0][row][column] / max(data_amp1_amp2[0][row]) * 2 - 1)
                    data_amp2_temp.append(data_amp1_amp2[1][row][column] / max(data_amp1_amp2[1][row]) * 2 - 1)
                data_amp1_amp2_normalisation[0].append(data_amp1_temp)
                data_amp1_amp2_normalisation[1].append(data_amp2_temp)
                n += 1

            data_1_phase.append(label_phase)  # 存储plane1的相位
            res.append((torch.tensor(data_amp1_amp2_normalisation), torch.tensor(data_1_phase)))

        self.res = res
        print(len(res))
        print(res[0][0].shape)
        print(res[0][1].shape)
        print(res[0][0][0].shape)
        print(res[0][0][1].shape)

    def __len__(self):    # 数据集的长度
        return len(self.res)

    def __getitem__(self, idx):    # 按照索引读取每个元素的具体内容
        data,label=self.res[idx][0],self.res[idx][1]
        return data,label

def data_split(data, rate):
    train_l = int(len(data) * rate)#计算训练集的长度，rate 是训练集占数据集总长度的比例。
    test_l = len(data) - train_l#计算测试集的长度，即总长度减去训练集长度。
    """打乱数据集并且划分"""
    train_set, test_set = torch.utils.data.random_split(data, [train_l, test_l])#使用 random_split 函数将数据集 data 随机划分为长度为 train_l 的训练集和长度为 test_l 的测试集。
    return train_set, test_set

data = MyDataset(r"E:\yy\ynet\train2000_240_12lambda_1.5")
train_set, test_set = data_split(data, 0.9)
# 加载数据集（训练集和测试集）
# 这里train_loader里含有500个batch，每个batch里是一个含两个元素的列表，第一个元素是tensor([100*2*8*160])，第二个是tensor([100*160])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)  # shuffle表示打乱数据
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
print("数据预处理完成")



# 定义模型组件
#残差层

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Identity() if stride == 1 and in_channels == out_channels else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out



# 两次卷积
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


# 下采样
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

# 上采样
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):      # 将x2与x1拼接
        x1=self.up(x1)
        # input is CHW
        diffY=torch.tensor([x2.size()[2]-x1.size()[2]])
        diffX=torch.tensor([x2.size()[3]-x1.size()[3]])

        x1=F.pad(x1,[diffX//2,diffX-diffX//2,
                     diffY//2,diffY-diffY//2])

        x=torch.cat([x2,x1],dim=1)
        return self.conv(x)

# 输出层
class OutConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(OutConv,self).__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size=1)

    def forward(self,x):
        return self.conv(x)


# 整合Y-net组件
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.canca1=ResidualBlock(64,64)
        self.down1 = Down(64, 128)
        self.canca2=ResidualBlock(128,128)
        self.down2 = Down(128, 256)
        self.canca3=ResidualBlock(256,256)
        self.down3 = Down(256, 256)
        self.up1 = Up(512, 256, bilinear)
        self.canca4=ResidualBlock(256,256)
        self.up2 = Up(384, 128, bilinear)
        self.canca5=ResidualBlock(128,128)
        self.up3 = Up(192, 64, bilinear)
        self.canca6=ResidualBlock(64,64)
        self.outc = OutConv(64, n_classes)

    def forward(self, a):
        a1 = self.inc(a)
        a2 = self.canca1(a1)
        a3 = self.down1(a2)
        a4 = self.canca2(a3)
        a5 = self.down2(a4)
        a6 = self.canca3(a5)
        a7 = self.down3(a6)


        x = self.up1(a7, a6)
        x = self.canca4(x)
        x = self.up2(x, a4)
        x = self.canca5(x)
        x = self.up3(x, a2)
        x = self.canca6(x)
        plane1_phase = self.outc(x)


        return plane1_phase

# 训练和记录输出函数封装
class Logger():
    def __init__(self, filename="log1.txt"):
        self.Terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.Terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def train_net_supervision(net,device,train_loader):
    # 定义SGD算法 可以更换为其他算法Adam等
    # optimizer = torch.optim.SGD(net.parameters(), lr=1e-5, weight_decay=1e-8, momentum=0.9)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    # 训练模式
    net.train()
    # 训练平均损失
    train_loss = 0.0
    # 按照batch_size开始训练
    for batch_index, (data,label) in enumerate(train_loader):
        # print(image)#开始一个循环，遍历 train_loader 中的每个批次。train_loader 是一个迭代器，每次迭代返回一个包含六个元素的元组 (data1, data2, plane1_amp, plane2_amp, label, _)。其中，data1 和 data2 是输入数据，label 是实际的相位标签，_ 是忽略的元素。
        optimizer.zero_grad()
        # 将数据拷贝到device中
        data = data.to(device=device, dtype=torch.float32)#将输入数据和标签复制到指定的设备（如 GPU），并将数据类型转换为 float32。
        label = label.to(device=device, dtype=torch.float32)
        # 使用网络参数，输出预测结果
        pred_plane1_phase= net(data)
        # 计算loss
        loss_supervised_mse = F.l1_loss(pred_plane1_phase, label)    # 与实际相位标签的mse损失
        # 更新参数
        # print("backward after")
        loss_supervised_mse.backward()#计算损失函数的梯度，并通过调用 backward() 方法在神经网络的参数上反向传播这些梯度。
        # print("back optim")
        optimizer.step()#使用优化器更新神经网络的参数。
        # 计算训练损失
        train_loss += loss_supervised_mse.item()#将当前批次的损失值加到 train_loss 中，以累积总的训练损失。
    train_loss /= len(train_loader.dataset)#计算平均训练损失，并将其添加到 train_loss_list 中，然后打印出来。train_loss 被除以训练数据集的大小，以得到平均损失值。
    train_loss_list.append(train_loss)
    print("train average loss:{}".format(train_loss))

def test_net_supervision(net,device,test_loader):
    # 训练模式
    net.eval()
    with torch.no_grad():  # 不会计算梯度，也不进行反向传播
        for batch_index, (data,label) in enumerate(test_loader):
            # 将数据拷贝到device中
            data = data.to(device=device, dtype=torch.float32)
            # 使用网络参数，输出预测结果
            pred_plane1_phase = net(data)
            # 计算loss
            loss_supervised_mse = F.mse_loss(pred_plane1_phase, label)  # 与实际相位标签的mse损失
            # 测试网络能力，获取预测rs和label
            np_pre = np.array(pred_plane1_phase.cpu())#将预测结果和实际标签从 GPU 复制到 CPU，并转换为 NumPy 数组。
            np_label=np.array(label.cpu())#np.array 将张量转换为 NumPy 数组。
            np.savetxt("U3_pre_plane1_phase_y.txt", np_pre.reshape(600,60))
            np.savetxt("U3_plane1_phase_y.txt",np_label.reshape(600,60))


def test(net,device,test_loader):
    # 测试模式
    net.eval()
    # 测试损失
    test_loss = 0.0
    with torch.no_grad():  # 不会计算梯度，也不进行反向传播
        for batch_index, (data,label) in enumerate(test_loader):
            # 将数据拷贝到device中
            data = data.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            # 使用网络参数，输出预测结果
            pred_plane1_phase = net(data)
            # 计算loss
            loss_supervised_mse = F.l1_loss(pred_plane1_phase, label)  # 与实际相位标签的mse损失
            test_loss+=loss_supervised_mse.item()
        test_loss /= len(test_loader.dataset)
        test_loss_list.append(test_loss)
        print("test average loss:{}\n".format(test_loss))
    return test_loss

# 加载网络，图片单通道，分类为1。
net=UNet(n_channels=2,n_classes=1)
# 将网络拷贝到device中
net.to(device=device)
# # 加载模型参数
# try:
#     net=torch.load("E:\YMH\\rough_trained_net\\best_model_unsupervision_Y-net.pth")
# except:
#     pass

# 调用方法
if __name__=="__main__":
    epoch_list = []  # 训练轮次
    train_loss_list = []
    test_loss_list = []
    test_loss_min = 100

    # 训练网络
    for i in range(1, epoch + 1):#开始一个循环，循环次数为 epoch。每次循环表示一个训练轮次
        epoch_list.append(i)
        print("train epoch:", i)
        train_net_supervision(net, device, train_loader)
        test_loss = test(net, device, test_loader)
        if test_loss<test_loss_min:
            test_loss_min=test_loss
            # if test_myloss < 1e-9:
            torch.save(net,"E:\\yy\\ynet\\model2.pth")
    print(test_loss_min)

    # # 测试网络
    # for i in range(1, 2):
    #     epoch_list.append(i)
    #     print("test epoch:", i)
    #     # train(net, device, train_loader, optimizer, i)
    #     test_myloss = test_net(net, device, test_loader)

    # 可视化loss
    plt.plot(epoch_list, train_loss_list, label='train_loss')
    plt.plot(epoch_list, test_loss_list, label='test_loss')
    plt.xlabel("Number of epoch")
    plt.ylabel("mse loss")
    plt.title("MSE Loss")
    plt.legend()
    plt.savefig("best_model_supervision_X-net_2000_Unet_y_fangxiang_mae.png")
    plt.show()
