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

        # ya4 = np.expand_dims(pywt.waverec(np.multiply(wave, [1, 0, 0, 0, 0]).tolist(), self.wavelet, axis=0),3)  #��4����Ʒ���
        # yd4 = np.expand_dims(pywt.waverec(np.multiply(wave, [0, 1, 0, 0, 0]).tolist(), self.wavelet, axis=0),3)  # 4-8hz�ع�С������ ���ɣ�
        # yd3 = np.expand_dims(pywt.waverec(np.multiply(wave, [0, 0, 1, 0, 0]).tolist(), self.wavelet, axis=0),3)  # 8-16hz�ع�С������ ���ɣ�
        # yd2 = np.expand_dims(pywt.waverec(np.multiply(wave, [0, 0, 0, 1, 0]).tolist(), self.wavelet, axis=0),3)  # 16-32hz�ع�С������ ���ɣ�
        # yd1 = np.expand_dims(pywt.waverec(np.multiply(wave, [0, 0, 0, 0, 1]).tolist(), self.wavelet, axis=0),3)  # 32-64hz�ع�С������ ���ɣ�

        ya4 = pywt.waverec(np.multiply(wave, [1, 0, 0, 0, 0]).tolist(), self.wavelet, axis=0)  #��4����Ʒ���
        yd4 = pywt.waverec(np.multiply(wave, [0, 1, 0, 0, 0]).tolist(), self.wavelet, axis=0)  # 4-8hz�ع�С������ ���ɣ�
        yd3 = pywt.waverec(np.multiply(wave, [0, 0, 1, 0, 0]).tolist(), self.wavelet, axis=0)  # 8-16hz�ع�С������ ���ɣ�
        yd2 = pywt.waverec(np.multiply(wave, [0, 0, 0, 1, 0]).tolist(), self.wavelet, axis=0)  # 16-32hz�ع�С������ ���ɣ�
        yd1 = pywt.waverec(np.multiply(wave, [0, 0, 0, 0, 1]).tolist(), self.wavelet, axis=0)  # 32-64hz�ع�С������ ���ɣ�

        # sp = torch.from_numpy(np.concatenate([ya4, yd4, yd3, yd2, yd1], axis=3)).cuda()
        sp = torch.from_numpy(np.concatenate([np.expand_dims(ya4, 3), np.expand_dims(yd4, 3), np.expand_dims(yd3, 3), np.expand_dims(yd2, 3), np.expand_dims(yd1, 3)], axis=3)).cuda(self.device)
        # sp = torch.from_numpy([ya4, yd4, yd3, yd2, yd1])

        # plt_x = [i for i in range(100)]
        #
        # plt.title('original time series')  # ����ͼ����
        # plt.xlabel('time')  # x�����
        # plt.plot(plt_x, data.cpu().detach().numpy()[0,:100,3])  # ��������ͼ��������ݵ㣬���õ�Ĵ�С
        # plt.show()
        #
        # plt.title('32-64hz')  # ����ͼ����
        # plt.xlabel('time')  # x�����
        # plt.plot(plt_x, yd1[0,:100,3])  # ��������ͼ��������ݵ㣬���õ�Ĵ�С
        # plt.show()
        #
        # plt.title('16-32hz')  # ����ͼ����
        # plt.xlabel('time')  # x�����
        # plt.plot(plt_x, yd2[0,:100,3])  # ��������ͼ��������ݵ㣬���õ�Ĵ�С
        # plt.show()
        #
        # plt.title('8-16hz')  # ����ͼ����
        # plt.xlabel('time')  # x�����
        # plt.plot(plt_x, yd3[0,:100,3])  # ��������ͼ��������ݵ㣬���õ�Ĵ�С
        # plt.show()
        #
        # plt.title('4-8hz')  # ����ͼ����
        # plt.xlabel('time')  # x�����
        # plt.plot(plt_x, yd4[0,:100,3])  # ��������ͼ��������ݵ㣬���õ�Ĵ�С
        # plt.show()
        #
        # plt.title('Approximate component')  # ����ͼ����
        # plt.xlabel('time')  # x�����
        # plt.plot(plt_x, ya4[0,:100,3])  # ��������ͼ��������ݵ㣬���õ�Ĵ�С
        # plt.show()

        return sp

