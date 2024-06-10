import numpy as np
import torch
import torch.nn as nn
import pywt
from utils.plotting import *

class Dwt(nn.Module):
    def __init__(self, configs, wavelet='db38', mode='symmetric'):
        super(Dwt, self).__init__()
        self.wavelet = wavelet
        self.mode = mode
        self.level = configs.d_level
        self.seq_len = configs.seq_len
        self.device = torch.device('cuda:{}'.format(configs.gpu))


    def forward(self, data):

        wave= pywt.wavedec(data.cpu().detach().numpy(), self.wavelet, self.mode, self.level, axis=0)

        # ya4 = np.expand_dims(pywt.waverec(np.multiply(wave, [1, 0, 0, 0, 0]).tolist(), self.wavelet, axis=0),3)  #第4层近似分量
        # yd4 = np.expand_dims(pywt.waverec(np.multiply(wave, [0, 1, 0, 0, 0]).tolist(), self.wavelet, axis=0),3)  # 4-8hz重构小波（θ 节律）
        # yd3 = np.expand_dims(pywt.waverec(np.multiply(wave, [0, 0, 1, 0, 0]).tolist(), self.wavelet, axis=0),3)  # 8-16hz重构小波（α 节律）
        # yd2 = np.expand_dims(pywt.waverec(np.multiply(wave, [0, 0, 0, 1, 0]).tolist(), self.wavelet, axis=0),3)  # 16-32hz重构小波（β 节律）
        # yd1 = np.expand_dims(pywt.waverec(np.multiply(wave, [0, 0, 0, 0, 1]).tolist(), self.wavelet, axis=0),3)  # 32-64hz重构小波（γ 节律）

        ya4 = pywt.waverec(np.multiply(wave, [1, 0, 0, 0, 0]).tolist(), self.wavelet, axis=0)  #第4层近似分量
        yd4 = pywt.waverec(np.multiply(wave, [0, 1, 0, 0, 0]).tolist(), self.wavelet, axis=0)  # 4-8hz重构小波（θ 节律）
        yd3 = pywt.waverec(np.multiply(wave, [0, 0, 1, 0, 0]).tolist(), self.wavelet, axis=0)  # 8-16hz重构小波（α 节律）
        yd2 = pywt.waverec(np.multiply(wave, [0, 0, 0, 1, 0]).tolist(), self.wavelet, axis=0)  # 16-32hz重构小波（β 节律）
        yd1 = pywt.waverec(np.multiply(wave, [0, 0, 0, 0, 1]).tolist(), self.wavelet, axis=0)  # 32-64hz重构小波（γ 节律）

        # sp = torch.from_numpy(np.concatenate([ya4, yd4, yd3, yd2, yd1], axis=3)).cuda()
        sp = torch.from_numpy(np.concatenate([np.expand_dims(ya4, 3), np.expand_dims(yd4, 3), np.expand_dims(yd3, 3), np.expand_dims(yd2, 3), np.expand_dims(yd1, 3)], axis=3)).cuda(self.device)
        # sp = torch.from_numpy([ya4, yd4, yd3, yd2, yd1])

        # plt_x = [i for i in range(100)]
        #
        # plt.title('original time series')  # 折线图标题
        # plt.xlabel('time')  # x轴标题
        # plt.plot(plt_x, data.cpu().detach().numpy()[0,:100,3])  # 绘制折线图，添加数据点，设置点的大小
        # plt.show()
        #
        # plt.title('32-64hz')  # 折线图标题
        # plt.xlabel('time')  # x轴标题
        # plt.plot(plt_x, yd1[0,:100,3])  # 绘制折线图，添加数据点，设置点的大小
        # plt.show()
        #
        # plt.title('16-32hz')  # 折线图标题
        # plt.xlabel('time')  # x轴标题
        # plt.plot(plt_x, yd2[0,:100,3])  # 绘制折线图，添加数据点，设置点的大小
        # plt.show()
        #
        # plt.title('8-16hz')  # 折线图标题
        # plt.xlabel('time')  # x轴标题
        # plt.plot(plt_x, yd3[0,:100,3])  # 绘制折线图，添加数据点，设置点的大小
        # plt.show()
        #
        # plt.title('4-8hz')  # 折线图标题
        # plt.xlabel('time')  # x轴标题
        # plt.plot(plt_x, yd4[0,:100,3])  # 绘制折线图，添加数据点，设置点的大小
        # plt.show()
        #
        # plt.title('Approximate component')  # 折线图标题
        # plt.xlabel('time')  # x轴标题
        # plt.plot(plt_x, ya4[0,:100,3])  # 绘制折线图，添加数据点，设置点的大小
        # plt.show()

        return sp

