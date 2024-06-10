import sys

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import functional as F

import os
import time
import cv2
import scienceplots

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('..//')
from exp.exp_basic import Exp_Basic

from data.data_loader import Dataset_Wg, Dataset_studWg, Dataset_AD
from models import PdMformer, PdMformerA, PdMformerB, Transformer, LSTM, GRU, ConvLSTM, TCN, FCN, STGCN, Autoformer, Fedformer, PatchTST, iTransformer, TimesNet, TimeMixer

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, auc

from utils.plotting import *
from utils.spot import SPOT

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):

        model_dict = {
            'PdMformer': PdMformer,
            'PdMformerA': PdMformerA,
            'PdMformerB': PdMformerB,
            'Transformer': Transformer,
            'LSTM': LSTM,
            'GRU': GRU,
            'ConvLSTM': ConvLSTM,
            'TCN': TCN,
            'FCN':FCN,
            'STGCN':STGCN,
            'Autoformer':Autoformer,
            'Fedformer':Fedformer,
            'PatchTST':PatchTST,
            'iTransformer':iTransformer,
            'TimesNet':TimesNet,
            'TimeMixer':TimeMixer        
        }

        model = model_dict[self.args.model].Model(self.args).float()

        # multi_gpu parallel train
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):

        args = self.args

        data_dict = {
            'welding_gun_55090573052':  Dataset_Wg,
            'bmw_welding_gun': Dataset_Wg,
            'bmw_stud_welding_gun_v3': Dataset_studWg,

            'SMD': Dataset_AD,
            'SWaT':Dataset_AD,
            'MSL': Dataset_AD,
            'SMAP': Dataset_AD,
            'NAB': Dataset_AD,
            'UCR': Dataset_AD,
        }

        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            ## TODO
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.pred_len],
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )

        print(flag, len(data_set))

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader


    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):

        criterion = nn.MSELoss(reduction = 'none')

        return criterion


    def vali(self, vali_data, vali_loader):
        self.model.eval()
        total_loss = []

        with torch.no_grad():
            for i, (batch_x, batch_x_mark, label_trues) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                label_trues = label_trues.to(self.device)

                # decoder input
                z = self.model(batch_x)
                z = z[2]

                #Error weight
                # weight = torch.tensor(([1.0, 2.0, 3.0, 5.0]))
                # loss_fct = nn.BCELoss(weight=weight)
                loss_fct = nn.BCELoss()

                output = torch.sigmoid(z)
                classific_loss = loss_fct(output.detach().cpu(), label_trues.detach().cpu())

                total_loss.append(classific_loss)

            total_loss = np.average(total_loss)

        self.model.train()

        return total_loss

    def train(self, setting, load = False):

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        self.model = self._build_model().to(self.device)

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
            print('Load model successful')

        else:
            path = os.path.join(self.args.checkpoints, setting)
            if not os.path.exists(path):
                os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        reconstruct_criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            attn_w1 = np.array([])
            attn_w2 = np.array([])
            attn_w3 = np.array([])

            attn_w1 = torch.Tensor(attn_w1)
            attn_w2 = torch.Tensor(attn_w2)
            attn_w3 = torch.Tensor(attn_w3)

            # batch_x_mark is timefeature of batch_x
            for i, (batch_x, batch_x_mark, label_trues) in enumerate(train_loader):

                iter_count += 1

                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device).permute(0, 1, 2)

                z = self.model(batch_x)

                # attn_w1 = torch.cat([attn_w1, z[3].reshape([z[3].shape[0] * z[3].shape[1], z[3].shape[2]]).detach().cpu()], 0)
                # attn_w2 = torch.cat([attn_w2, z[4].reshape([z[4].shape[0] * z[4].shape[1], z[4].shape[2]]).detach().cpu()], 0)
                # attn_w3 = torch.cat([attn_w3, z[5].reshape([z[5].shape[0] * z[5].shape[1], z[5].shape[2]]).detach().cpu()], 0)

                # weight = torch.tensor(([1.0, 2.0, 3.0, 5.0])).to(self.device)
                # loss_fct = nn.BCELoss(weight=weight)
                loss_fct = nn.BCELoss()
                output = torch.sigmoid(z[2])

                classific_loss = loss_fct(output, label_trues.to(self.device))

                # phase 1 reconstruction
                r_reconstruct_loss = reconstruct_criterion(z[0], batch_x)
                # phase 2 reconstruction
                ad_reconstruct_loss = reconstruct_criterion(z[1], batch_x)
                # adversarial training loss
                regression_loss = (1 - epoch / (self.args.train_epochs)) * reconstruct_criterion(z[0], batch_x) + (1 - 1.0 * (epoch / (self.args.train_epochs))) * reconstruct_criterion(z[1], batch_x)
                Adversarial_loss = - 1.0 * ((epoch / (self.args.train_epochs))) * reconstruct_criterion(z[1], batch_x)

                # ##  TODO phase 1 与 2 的重构比较

                # if label_trues[0,0] == 0:
                
                #     x = [i for i in range(600)]
                #     phase1 = z[0].detach().cpu()
                #     phase2 = z[1].detach().cpu()
                #     batch_true = batch_x.detach().cpu()
                
                #     plt.title('reconstruction')  # 折线图标题
                #     plt.xlabel('time')  # x轴标题
                #     plt.ylabel('sensor data')  # y轴标题
                #     plt.plot(x, batch_true[0, :600, 7], linewidth=0.4)  # 绘制折线图，添加数据点，设置点的大小
                #     plt.plot(x, phase1[0, :600, 7], linewidth=0.4)  # 绘制折线图，添加数据点，设置点的大小
                #     plt.plot(x, phase2[0, :600, 7], linewidth=0.4)  # 绘制折线图，添加数据点，设置点的大小
                #     plt.legend(['trues', 'phase 1', 'phase 2'],bbox_to_anchor=(1.0, 0), loc=3, borderaxespad=0)
                #     nm = 'reconstruction' + str(7) + '.png'
                #     plt.show()
                #     plt.close()

                r_reconstruct_loss = torch.mean(r_reconstruct_loss)
                ad_reconstruct_loss = torch.mean(ad_reconstruct_loss)
                min_loss = torch.mean(regression_loss)
                max_loss = torch.mean(Adversarial_loss)

                train_loss.append(classific_loss.detach().cpu().numpy())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | classific_loss: {2:.7f} | r_regression_loss: {3:.7f} | ad_regression_loss: {4:.7f} | min_loss: {5:.7f} | max_loss: {6:.7f}"
                          .format(i + 1, epoch + 1, classific_loss.item(), r_reconstruct_loss.item(), ad_reconstruct_loss.item(), min_loss, max_loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(min_loss).backward(retain_graph=True)
                    scaler.scale(max_loss).backward(retain_graph=True)
                    scaler.scale(classific_loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    # min_loss.backward(retain_graph=True)
                    # max_loss.backward(retain_graph=True)
                    classific_loss.backward()
                    # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader)
            test_loss = self.vali(test_data, test_loader)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

            # self.draw_features(5, 100, attn_w1[:100,:],'attn_w1_Epoch' + str(epoch) + '.png')
            # self.draw_features(5, 100, attn_w2[:100, :],'attn_w2_Epoch' + str(epoch) + '.png')
            # self.draw_features(5, 100, attn_w3[:100, :],'attn_w3_Epoch' + str(epoch) + '.png')

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def draw_features(self, width, height, x, savename):
        tic = time.time()
        fig = plt.figure(figsize=(16, 16))
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)

        plt.axis('off')
        img = np.array(x)
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255  # float在[0，1]之间，转换成0-255
        img = img.astype(np.uint8)  # 转成unit8
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map
        img = img[:, :, ::-1]  # 注意cv2（BGR）和matplotlib(RGB)通道是相反的
        plt.imshow(img)
        fig.savefig(savename, dpi=1000)
        fig.clf()
        plt.close()


    def test(self, setting, load=False):
        test_data, test_loader = self._get_data(flag='test')

        if load:
            self.model = self._build_model().to(self.device)
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = os.path.join(self.args.checkpoints, setting) + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()

        preds = []
        trues = []

        # sensor_names_units = {
        #     0: ('C_Cylinder_force', 'N'),  # 假设单位为牛顿
        #     1: ('C_Differential_pressure', 'Pa'),  # 假设单位为帕斯卡
        #     2: ('W_Welding_point_count', 'points'),  # 计数，没有单位
        #     3: ('W_position_count', 'points'),  # 同上，计数
        #     4: ('in_Counterbalance_pressure', 'Pa'),  # 假设单位为帕斯卡
        #     5: ('in_Electrode_force', 'N'),  # 假设单位为牛顿
        #     6: ('in_Electrode_position', 'mm'),  # 假设单位为毫米
        #     7: ('in_Sheet_thickness', 'mm'),  # 假设单位为毫米
        #     8: ('in_Velocity', 'm/s'),  # 假设单位为米/秒
        #     9: ('in_force_build_up', 'N'),  # 假设单位为牛顿
        #     10: ('out_Cap_offset', 'mm'),  # 假设单位为毫米
        #     11: ('out_Electrode_force', 'N'),  # 假设单位为牛顿
        #     12: ('out_Electrode_position', 'mm'),  # 假设单位为毫米
        #     13: ('out_force_build_up', 'N'),  # 假设单位为牛顿
        # }

        start_time = time.time()

        for i, (batch_x, batch_x_mark, label_trues) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            label_trues = label_trues.to(self.device)

            z = self.model(batch_x)         

            preds.append(z[2].detach().cpu().numpy())
            trues.append(label_trues.detach().cpu().numpy())

            # ##  TODO phase 1 与 2 的重构比较
            # for m in range(label_trues.shape[0]):
                
            #     if label_trues[m,0] == 0:                     
                
            #         x = [l for l in range(50)]
            #         phase1 = z[0].detach().cpu()
            #         phase2 = z[1].detach().cpu()
            #         batch_true = batch_x.detach().cpu()

                    # for k in range(batch_true.shape[2]):
                    #     sensor_name, unit = sensor_names_units[k]  # 获取传感器名称和单位
                    #     plt.title(f'Comparison between Normal and Abnormal Time Series', fontsize=10)  # 折线图标题
                    #     plt.xlabel('Time / s', fontsize=8)
                    #     plt.ylabel(f'{sensor_name} \n Sensor Data', fontsize=8)  # 使用传感器单位更新y轴标题

                    #     # 接下来是你的绘图代码...
                    #     plt.plot(x, batch_true[0, -200:-150, k], linewidth=0.5)  # 稍微增加线宽以提升可读性
                    #     plt.plot(x, phase2[0, -200:-150, k], linewidth=0.5)
                    #     plt.legend(['Abnormal', 'Normal'], bbox_to_anchor=(1.0, 0.5), loc='center left', borderaxespad=0, fontsize=9)
                    #     nm = f'C:/PdMPic/{sensor_name}/reconstruction_{4*i+m}.png'

                    #     # 在保存图片之前创建目录
                    #     os.makedirs(os.path.dirname(nm), exist_ok=True)

                    #     plt.tight_layout()
                    #     plt.xticks(fontsize=9)
                    #     plt.yticks(fontsize=9)
                    #     plt.grid(True)  # 增加网格线以提升可读性
                    #     plt.savefig(nm, dpi=300, bbox_inches='tight')
                    #     plt.close()

        preds = np.array(preds)
        trues = np.array(trues)

        preds = preds.reshape(preds.shape[0]*preds.shape[1], preds.shape[2])
        trues = trues.reshape(trues.shape[0]*trues.shape[1], trues.shape[2])

        print(trues.shape)

        inference_time = time.time() - start_time

        # # 添加计算AUC-ROC和混淆矩阵
        # auc_score = roc_auc_score(trues, preds, multi_class='ovo') # 对于多类问题使用'ovo'或'ovr'
  
        
        # # 输出混淆矩阵和AUC-ROC得分
        # print(f'AUC-ROC: {auc_score}')
        
        # 原有的报告保存代码...
        print(f"Inference time: {inference_time} seconds")

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        probs = torch.nn.functional.sigmoid(torch.tensor(preds))

        # 将tensor转换为NumPy数组
        numpy_array = probs.numpy()

        # 将数组保存为CSV文件
        filename = f"{self.args.model}.csv"
        np.savetxt(filename, numpy_array, delimiter=",")
        numpy_array = trues
        np.savetxt('true.csv', numpy_array, delimiter=",")

        # ROC曲线计算和绘制
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(trues.shape[1]):  # 假设trues是one-hot编码
            fpr[i], tpr[i], _ = roc_curve(trues[:, i], probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # 绘制所有类别的ROC曲线
        plt.figure(figsize=(7, 6))
        for i in range(trues.shape[1]):
            plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic for {setting}')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join('roc_curve.png'))
        plt.close()  # 关闭图表，防止在笔记本中显示

        cm = confusion_matrix(trues.argmax(axis=1), probs.argmax(axis=1)) # 对于多类问题，使用argmax获取类别标签
        print('Confusion Matrix:')
        print(cm)

        pred = (np.array(probs) > 0.5).astype(int)

        report = classification_report(trues, pred, digits=5)

        print('report:{}'.format(report))

        np.save(folder_path + 'report.npy', report)

        return


    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()

        preds = []

        for i, (batch_x, batch_x_mark) in enumerate(pred_loader):
            batch_x = batch_x.float().to(self.device)

            z = self.model(batch_x)
            z = z[2]

            preds.append(z.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return

