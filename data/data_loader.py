import os
import numpy as np
import pandas as pd

import sys

import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler

import sys
sys.path.append('..//')
from utils.tools import StandardScaler
from utils.timefeatures import time_features
import utils.global_var

from utils.plotting import *

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

class Dataset_Wg(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='welding_gun_55090573052.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='s', cols=None):

        # size [seq_len, pred_len]
        # info

        if size == None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path

        self.__read_data__()

    def __read_data__(self):

        self.scaler = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date')

        print(cols)

        df_raw = df_raw[['date']+cols+[self.target]]


        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''

        ## Transfer Error to num
        temp_label = df_raw[self.target]
        temp_label = pd.Series(temp_label, dtype="category")
        class_names = temp_label.cat.categories

        utils.global_var.set_value('class_names', class_names)

        df_raw = df_raw.groupby('gun_no')

        num_flag = 0

        for gun_no in df_raw.size().index:

                cols_data = df_raw.get_group(gun_no).columns[1:-1]
                cols_label = df_raw.get_group(gun_no).columns[-1]

                df_data = df_raw.get_group(gun_no)[cols_data]
                df_label = df_raw.get_group(gun_no)[cols_label]

                df_label = pd.Series(df_label, dtype="category")
                df_label = pd.DataFrame(df_label.cat.codes, dtype=np.int8)

                train_dataset, train_labels = self.create_dataset(df_data, df_label, seq_len=self.seq_len, predict_time=self.pred_len, interval=int(1200))

                # X_train, X_val, X_test = self.split_data(train_dataset)
                # y_train, y_val, y_test = self.split_data(train_labels)

                X_train, X_test, y_train, y_test = train_test_split(train_dataset, train_labels, test_size=0.25, random_state=17)
                X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.4, random_state=17)

                if num_flag == 0:
                    self.train_x = X_train
                    self.train_y = y_train
                    self.val_x = X_val
                    self.val_y = y_val
                    self.test_x = X_test
                    self.test_y = y_test

                else:
                    self.train_x = np.concatenate((self.train_x, X_train), axis=0)
                    self.train_y = np.concatenate((self.train_y, y_train), axis=0)
                    self.val_x = np.concatenate((self.val_x, X_val), axis=0)
                    self.val_y = np.concatenate((self.val_y, y_val), axis=0)
                    self.test_x = np.concatenate((self.test_x, X_test), axis=0)
                    self.test_y = np.concatenate((self.test_y, y_test), axis=0)

                num_flag += 1

        train_stamp =self.train_x[:, :, 0]
        val_stamp = self.val_x[:, :, 0]
        test_stamp = self.test_x[:, :, 0]

        self.train_x = np.array(self.train_x[:, :, 1:], dtype=np.float64)

        if self.scale:
            self.scaler.fit(self.train_x)
            self.train_x = self.scaler.transform(self.train_x)
            self.val_x = self.scaler.transform(np.array(self.val_x[:, :, 1:], dtype=np.float64))
            self.test_x = self.scaler.transform(np.array(self.test_x[:, :, 1:], dtype=np.float64))

        if self.set_type == 0:
            self.data_x = self.train_x
            self.data_y = self.train_y
            self.data_stamp = train_stamp
        elif self.set_type == 1:
            self.data_x = self.val_x
            self.data_y = self.val_y
            self.data_stamp = val_stamp
        elif self.set_type == 2:
            self.data_x = self.test_x
            self.data_y = self.test_y
            self.data_stamp = test_stamp

        print(self.data_x.shape)
        print(self.data_y.shape)

    def __getitem__(self, index):

        s_end = self.seq_len

        seq_x = self.data_x[index][:s_end,:]
        seq_x_mark = time_features(self.data_stamp[index][:s_end], timeenc=self.timeenc, freq=self.freq)

        trues = torch.Tensor(np.array(self.data_y[index]))
        return seq_x, seq_x_mark, trues

    def __len__(self):
        return len(self.data_x)-10

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    # 构建特征数据集
    def create_dataset(self, x, y, seq_len=600, predict_time=600,  interval=600):
        features = []
        targets = []

        for i in range(0, len(x) - seq_len - predict_time, interval):


            data = x[i:i + seq_len + predict_time]

            multi_cls_lab = np.zeros((len(utils.global_var.get_value('class_names'))), np.float32)

            label_value = np.unique(y.iloc[i + seq_len:i + seq_len + predict_time])

            # ##TODO
            
            # if len(label_value) > 1:
            
            #     xx = y.iloc[i + seq_len  : i + seq_len + 1200]
            #     xxx_y = data.iloc[seq_len : seq_len + 1200, 1].reset_index(drop=True)
            #     xxx_x = data.iloc[seq_len : seq_len + 1200, 1].reset_index(drop=True)
            
            #     xx = xx.squeeze()
            
            #     for k, xxx in enumerate(xx):
            
            #         if xxx == 0:
            #             xxx_y[k] = None
            
            
            #     plt_x = [i for i in range(1200)]
            
            #     plt.title('reconstruction')  # 折线图标题
            #     plt.xlabel('time')  # x轴标题
            #     plt.ylabel('sensor data')  # y轴标题
            #     plt.plot(plt_x, xxx_x, linewidth=0.4)  # 绘制折线图，添加数据点，设置点的大小
            #     plt.plot(plt_x, xxx_y, linewidth=0.4)  # 绘制折线图，添加数据点，设置点的大小
            #     plt.show()

            if 0 in label_value and len(label_value) == 1:
                multi_cls_lab[0] = 1.0
            else:
                for num in label_value:

                    multi_cls_lab[int(num)] = 1.0
                multi_cls_lab[0] = 0.0

            label = multi_cls_lab

            features.append(data)
            targets.append(label)

        return np.array(features), np.array(targets)
    
    def split_data(self, df_raw):
        n = df_raw.shape[0]
        train_df = df_raw[:int(n*0.5)]
        val_df = df_raw[int(n*0.5):int(n*0.6)]
        test_df = df_raw[int(n*0.6):]

        return train_df, val_df, test_df


class Dataset_studWg(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='welding_gun_55090573052.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='s', cols=None):

        # size [seq_len, pred_len]
        # info

        if size == None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path

        self.__read_data__()

    def __read_data__(self):

        self.scaler = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns); cols.remove(self.target); cols.remove('enqueuedTime'); cols.remove('approximateTimes'); cols.remove('line'); cols.remove('studId')

        print(cols)

        df_raw = df_raw[['approximateTimes']+['studId']+cols+[self.target]]


        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''

        ## Transfer Error to num
        df_raw[self.target] =df_raw[self.target].astype(str)
        temp_label = df_raw[self.target]
        temp_label = pd.Series(temp_label, dtype="category")
        class_names = temp_label.cat.categories

        utils.global_var.set_value('class_names', class_names)

        df_raw = df_raw.groupby('studId')

        num_flag = 0

        for gun_no in df_raw.size().index:
                if df_raw.get_group(gun_no).shape[0] >= 10000:

                    cols_data = df_raw.get_group(gun_no).columns[0:-1]
                    cols_label = df_raw.get_group(gun_no).columns[-1]

                    df_data = df_raw.get_group(gun_no)[cols_data]
                    df_label = df_raw.get_group(gun_no)[cols_label]

                    df_label = pd.Series(df_label, dtype="category")
                    df_label = pd.DataFrame(df_label.cat.codes, dtype=np.int8)
                    train_dataset, train_labels = self.create_dataset(df_data, df_label, seq_len=self.seq_len, predict_time=self.pred_len, interval=self.seq_len)

                    X_train, X_test, y_train, y_test = train_test_split(train_dataset, train_labels, test_size=0.25, random_state=17)

                    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.4, random_state=17)


                    if num_flag == 0:
                        self.train_x = X_train
                        self.train_y = y_train
                        self.val_x = X_val
                        self.val_y = y_val
                        self.test_x = X_test
                        self.test_y = y_test

                    else:
                        self.train_x = np.concatenate((self.train_x, X_train), axis=0)
                        self.train_y = np.concatenate((self.train_y, y_train), axis=0)
                        self.val_x = np.concatenate((self.val_x, X_val), axis=0)
                        self.val_y = np.concatenate((self.val_y, y_val), axis=0)
                        self.test_x = np.concatenate((self.test_x, X_test), axis=0)
                        self.test_y = np.concatenate((self.test_y, y_test), axis=0)

                    num_flag += 1

        train_stamp =self.train_x[:, :, 0]
        val_stamp = self.val_x[:, :, 0]
        test_stamp = self.test_x[:, :, 0]

        self.train_x = np.array(self.train_x[:, :, 2:], dtype=np.float64)


        if self.scale:
            self.scaler.fit(self.train_x)
            self.train_x = self.scaler.transform(self.train_x)
            self.val_x = self.scaler.transform(np.array(self.val_x[:, :, 2:], dtype=np.float64))
            self.test_x = self.scaler.transform(np.array(self.test_x[:, :, 2:], dtype=np.float64))

        if self.set_type == 0:
            self.data_x = self.train_x
            self.data_y = self.train_y
            self.data_stamp = train_stamp
        elif self.set_type == 1:
            self.data_x = self.val_x
            self.data_y = self.val_y
            self.data_stamp = val_stamp
        elif self.set_type == 2:
            self.data_x = self.test_x
            self.data_y = self.test_y
            self.data_stamp = test_stamp

        print(self.data_x.shape)
        print(self.data_y.shape)

    def __getitem__(self, index):

        s_end = self.seq_len

        seq_x = self.data_x[index][:s_end,:]
        seq_x_mark = time_features(self.data_stamp[index][:s_end], timeenc=self.timeenc, freq=self.freq)

        trues = torch.Tensor(np.array(self.data_y[index]))
        return seq_x, seq_x_mark, trues

    def __len__(self):
        return len(self.data_x)-10

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    # 构建特征数据集
    def create_dataset(self, x, y, seq_len=600, predict_time=600,  interval=600):
        features = []
        targets = []

        for i in range(0, len(x) - seq_len - predict_time, interval):


            data = x[i:i + seq_len + predict_time]

            multi_cls_lab = np.zeros((len(utils.global_var.get_value('class_names'))), np.float32)

            label_value = np.unique(y.iloc[i + seq_len:i + seq_len + predict_time])


            if 0 in label_value and len(label_value) == 1:
                multi_cls_lab[0] = 1.0
            else:
                for num in label_value:

                    multi_cls_lab[int(num)] = 1.0
                multi_cls_lab[0] = 0.0

            label = multi_cls_lab

            features.append(data)
            targets.append(label)

        return np.array(features), np.array(targets)

class Dataset_AD(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='MS', data_path='bmw_welding_gun.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
    def __read_data__(self):
        folder = os.path.join(self.root_path, self.data_path)
        print(folder)
        if not os.path.exists(folder):
            raise Exception('Processed Data not found.')
        loader = []
        for flag in ['train', 'test', 'labels']:
            if self.data_path == 'SMD': file = 'machine-1-1_' + flag
            if self.data_path == 'SMAP': file = 'P-1_' + flag
            if self.data_path == 'MSL': file = 'C-1_' + flag
            if self.data_path == 'UCR': file = '136_' + flag
            if self.data_path == 'NAB': file = 'ec2_request_latency_system_failure_' + flag
            if self.data_path == 'SWaT': file = flag
            if flag == 'train':
                self.data_x = np.load(os.path.join(folder, f'{file}.npy'))
            if flag == 'test':
                self.data_y = np.load(os.path.join(folder, f'{file}.npy'))
            if flag == 'labels':
                self.data_stamp = np.load(os.path.join(folder, f'{file}.npy'))

        num_train = int(len(self.data_x)*0.7)

        self.data_train = self.data_x[:num_train]
        self.data_vali = self.data_x[num_train:]

    def __getitem__(self, index):

        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len


        if self.set_type == 0:
            seq_x = self.data_train[s_begin:s_end]
            seq_y = self.data_train[r_begin:r_end]

            seq_x_mark = self.data_stamp[0]
            seq_y_mark = self.data_stamp[0]
            trues = self.data_stamp[0]

        if self.set_type == 1:
            seq_x = self.data_vali[s_begin:s_end]
            seq_y = self.data_vali[r_begin:r_end]

            seq_x_mark = self.data_stamp[0]
            seq_y_mark = self.data_stamp[0]
            trues = self.data_stamp[0]

        if self.set_type == 2:
            seq_x = self.data_y[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
            trues = self.data_stamp[r_begin:r_end]

            seq_x_mark = self.data_stamp[0]
            seq_y_mark = self.data_stamp[0]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, trues

    def __len__(self):
        if self.set_type == 0:
            return len(self.data_train) - self.seq_len - self.pred_len

        if self.set_type == 1:
            return len(self.data_vali) - self.seq_len - self.pred_len

        if self.set_type == 2:
            return len(self.data_y) - self.seq_len - self.pred_len