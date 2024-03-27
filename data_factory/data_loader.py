import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle


class PSMSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, 1:]

        data = np.nan_to_num(data)

        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')

        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test

        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/MSL_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/MSL_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/MSL_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMAP_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMAP_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/SMAP_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMD_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMD_test.npy")
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(data_path + "/SMD_test_label.npy")

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])



class Dataset_vitaldb(Dataset):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        import sys
        import yaml
        from random import randint
        sys.path.append('/home/mjh319/workspace/3_hypotension_detection/3_hypo')
        sys.path.append('/home/mjh319/workspace/3_hypotension_detection/3_hypo/models')
        from datasets.dataset import Make_hypo_dataset

        config_path = '/home/mjh319/workspace/_hypo/4_hypo/config/0916_time.yml'
        opt = yaml.load(open(config_path), Loader=yaml.FullLoader)
        # opt.update(vars(self.args))

        dfcases = pd.read_csv("https://api.vitaldb.net/cases")

        opt['invasive'] = True
        opt['multi'] = True
        opt['pred_lag'] = 300
        opt['features'] = 'none'

        # random_key = randint(0, 100000)
        random_key = 777

        loader = Make_hypo_dataset(opt, random_key,dfcases)
        self.dataset_list_train = loader["train"].dataset.dataset_list_
        self.transform_train = loader["train"].dataset.transform
        self.dataset_list_test = loader["test"].dataset.dataset_list_
        self.transform_test = loader["test"].dataset.transform
        self.dataset_list_val = loader["valid"].dataset.dataset_list_
        self.transform_val = loader["valid"].dataset.transform
        self.feature_keys = loader["train"].dataset.feature_keys

        downsample_factor = 30  # 예시로 10개의 데이터 포인트를 평균화하여 다운샘플링
        

        for key in self.feature_keys:
            tp = self.dataset_list_train[key]
            downsampled_time_series = []
            targets = [] 
            # 각 시계열에서 다운샘플링 진행
            for i in range(tp.shape[0]):  # 시계열 개수에 대해 루프를 돕니다.
                downsampled_segment = np.mean(tp[i].reshape(-1, downsample_factor), axis=1)
                if self.dataset_list_train['target'][i] == 0:
                    downsampled_time_series.append(downsampled_segment)
                    targets.append(self.dataset_list_train['target'][i] )

            # 리스트를 넘파이 배열로 변환
            downsampled_time_series = np.stack(downsampled_time_series)
            self.dataset_list_train[key] = downsampled_time_series
        targets = np.stack(targets)
        self.dataset_list_train['target'] = targets

        
        for key in self.feature_keys:
            tp = self.dataset_list_test[key]
            downsampled_time_series = []
            # 각 시계열에서 다운샘플링 진행
            for i in range(tp.shape[0]):  # 시계열 개수에 대해 루프를 돕니다.
                downsampled_segment = np.mean(tp[i].reshape(-1, downsample_factor), axis=1)
                downsampled_time_series.append(downsampled_segment)

            # 리스트를 넘파이 배열로 변환
            downsampled_time_series = np.stack(downsampled_time_series)
            self.dataset_list_test[key] = downsampled_time_series

        for key in self.feature_keys:
            tp = self.dataset_list_val[key]
            downsampled_time_series = []
            # 각 시계열에서 다운샘플링 진행
            for i in range(tp.shape[0]):  # 시계열 개수에 대해 루프를 돕니다.
                downsampled_segment = np.mean(tp[i].reshape(-1, downsample_factor), axis=1)
                downsampled_time_series.append(downsampled_segment)

            # 리스트를 넘파이 배열로 변환
            downsampled_time_series = np.stack(downsampled_time_series)
            self.dataset_list_val[key] = downsampled_time_series


    def __getitem__(self, index):
        if self.mode == "train":
            for idx, feature in enumerate(self.feature_keys):
                if idx == 0:
                    data_ = self.transform_train[feature](np.expand_dims(self.dataset_list_train[feature][index,:], axis=0))
                else:
                    data_ = torch.cat([data_,
                                        self.transform_train[feature](np.expand_dims(self.dataset_list_train[feature][index,:], axis=0))], dim=0) 
            return np.array(data_.squeeze().transpose(1,0).to(torch.float32)), np.array(np.float32(self.dataset_list_train['target'][index]))
        elif (self.mode == 'val'):
            for idx, feature in enumerate(self.feature_keys):
                if idx == 0:
                    data_ = self.transform_val[feature](np.expand_dims(self.dataset_list_val[feature][index,:], axis=0))
                else:
                    data_ = torch.cat([data_,
                                        self.transform_val[feature](np.expand_dims(self.dataset_list_val[feature][index,:], axis=0))], dim=0) 
            return np.array(data_.squeeze().transpose(1,0).to(torch.float32)), np.array(np.float32(self.dataset_list_val['target'][index]))
        elif (self.mode == 'test'):
            for idx, feature in enumerate(self.feature_keys):
                if idx == 0:
                    data_ = self.transform_test[feature](np.expand_dims(self.dataset_list_test[feature][index,:], axis=0))
                else:
                    data_ = torch.cat([data_,
                                        self.transform_test[feature](np.expand_dims(self.dataset_list_test[feature][index,:], axis=0))], dim=0) 
            return np.array(data_.squeeze().transpose(1,0).to(torch.float32)), np.array(np.float32(self.dataset_list_test['target'][index]))
        else:
            for idx, feature in enumerate(self.feature_keys):
                if idx == 0:
                    data_ = self.transform_test[feature](np.expand_dims(self.dataset_list_test[feature][index,:], axis=0))
                else:
                    data_ = torch.cat([data_,
                                        self.transform_test[feature](np.expand_dims(self.dataset_list_test[feature][index,:], axis=0))], dim=0) 
            return np.array(data_.squeeze().transpose(1,0).to(torch.float32)), np.array(np.float32(self.dataset_list_test['target'][index]))


    def __len__(self):
        if self.mode == "train":
            return len(self.dataset_list_train['target'])
        elif (self.mode == 'val'):
            return len(self.dataset_list_val['target'])
        elif (self.mode == 'test'):
            return len(self.dataset_list_test['target'])
        else:
            return len(self.dataset_list_test['target'])

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', dataset='KDD'):
    if (dataset == 'SMD'):
        dataset = SMDSegLoader(data_path, win_size, step, mode)
    elif (dataset == 'MSL'):
        dataset = MSLSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'SMAP'):
        dataset = SMAPSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'PSM'):
        dataset = PSMSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'vitaldb'):
        dataset = Dataset_vitaldb(data_path, win_size, 1, mode)
    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader
