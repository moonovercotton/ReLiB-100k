import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from glob import glob
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# # ========== Unified Dataset 定义 ==========
# class BatteryDataset(Dataset):
#     def __init__(self, file_path, flag, transform=None):
#         """
#         Args:
#             file_path: pkl 文件路径
#             flag: contrastive 返回对比学习数据 (view1, view2, label(soh), capacity)
#                   cap_regression 返回单容量监督数据 (feature, label(max_capacity), capacity)
#                   soh_regression 返回混合容量监督数据 (feature, label(soh), capacity)
#             transform: 数据增强方法 (用于对比学习)，默认包含噪声+遮挡
#         """
#         self.samples = []
#         self.flag = flag
#         self.transform = transform if transform is not None else self.default_transform

#         with open(file_path, 'rb') as f:
#             data = pickle.load(f)
#             for sample in data:
#                 voltage_curve = sample['voltage_curve']
#                 current_curve = sample['current_curve']
#                 max_capacity = sample['max_capacity']    # 实际容量
#                 capacity = sample['capacity']    # 额定容量
#                 soh = sample['soh']     # 健康状态

#                 feature = np.stack((voltage_curve, current_curve), axis=1)  # [seq_len, 2]

#                 if self.flag == 'cap_regression':
#                     label = max_capacity
#                 else:
#                     label = soh

#                 self.samples.append((feature, label, capacity))

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         feature, label, capacity = self.samples[idx]

#         if self.flag == 'contrastive':
#             # 生成两个视图
#             view1 = self.transform(feature)
#             view2 = self.transform(feature)

#             view1 = torch.tensor(view1, dtype=torch.float32)
#             view2 = torch.tensor(view2, dtype=torch.float32)
#             label = torch.tensor(label, dtype=torch.float32)
#             capacity = torch.tensor(capacity, dtype=torch.float32)

#             return view1, view2, label, capacity
#         elif self.flag == 'cap_regression' or self.flag == 'soh_regression':
#             feature = torch.tensor(feature, dtype=torch.float32)
#             label = torch.tensor(label, dtype=torch.float32)
#             capacity = torch.tensor(capacity, dtype=torch.float32)

#             return feature, label, capacity

#     def default_transform(self, x):
#         """默认增强：高斯噪声 + 随机时间遮挡"""
#         x_aug = x.copy()

#         # 加噪声
#         noise = x * np.random.normal(0, 0.01, size=x.shape)
#         x_aug += noise

#         # 遮挡随机一段时间序列
#         if x.shape[0] > 10:
#             start = np.random.randint(0, x.shape[0] - 5)
#             x_aug[start:start+5, :] = 0  # 掩码5个时间步

#         return x_aug
    

# 加入归一化
class BatteryDataset(Dataset):
    def __init__(self, file_path, flag, transform=None):
        self.samples = []
        self.flag = flag
        self.transform = transform if transform is not None else self.default_transform

        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            for sample in data:
                voltage_curve = sample['voltage_curve']
                current_curve = sample['current_curve']
                max_capacity = sample['max_capacity']    # 实际容量
                capacity = sample['capacity']            # 额定容量
                soh = sample['soh']                      # 健康状态

                feature = np.stack((voltage_curve, current_curve), axis=1)  # [seq_len, 2]

                if self.flag == 'cap_regression':
                    label = max_capacity
                else:
                    label = soh

                self.samples.append((feature, label, capacity))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feature, label, capacity = self.samples[idx]

        # # ====== 每个样本单独 Min-Max 归一化 ======
        # feature_min = feature.min(axis=0)
        # feature_max = feature.max(axis=0)
        # feature = (feature - feature_min) / (feature_max - feature_min + 1e-8)

        # ====== 每个样本单独标准化 ======
        feature_mean = feature.mean(axis=0)
        feature_std = feature.std(axis=0)
        feature = (feature - feature_mean) / (feature_std + 1e-8)

        # ====== 对比学习视图 ======
        if self.flag == 'contrastive':
            view1 = self.transform(feature)
            view2 = self.transform(feature)

            view1 = torch.tensor(view1, dtype=torch.float32)
            view2 = torch.tensor(view2, dtype=torch.float32)
            label = torch.tensor(label, dtype=torch.float32)
            capacity = torch.tensor(capacity, dtype=torch.float32)
            return view1, view2, label, capacity

        # ====== 回归任务 ======
        elif self.flag in ['cap_regression', 'soh_regression']:
            feature = torch.tensor(feature, dtype=torch.float32)
            label = torch.tensor(label, dtype=torch.float32)
            capacity = torch.tensor(capacity, dtype=torch.float32)
            return feature, label, capacity

    def default_transform(self, x):
        """默认增强：高斯噪声 + 随机时间遮挡"""
        x_aug = x.copy()
        # 高斯噪声，标准差为样本值的 1%
        noise = x * np.random.normal(0, 0.01, size=x.shape)
        x_aug += noise
        # 随机时间遮挡
        if x.shape[0] > 10:
            start = np.random.randint(0, x.shape[0] - 5)
            x_aug[start:start+5, :] = 0
        return x_aug
