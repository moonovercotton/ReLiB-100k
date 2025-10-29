import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_dim=2, seq_len=512, num_filters=[64, 128, 256], kernel_size=3, dropout=0.2):
        """
        input_dim: 每个时间步的特征数
        seq_len: 序列长度（可用于计算全连接层输入）
        num_filters: 每层卷积的通道数
        """
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, num_filters[0], kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(num_filters[0], num_filters[1], kernel_size, padding=kernel_size//2)
        self.conv3 = nn.Conv1d(num_filters[1], num_filters[2], kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        self.num_filters = num_filters

        # 全连接层
        self.fc = nn.Linear(num_filters[-1], 1)

    def forward(self, x):
        """
        输入:
            x: [B, seq_len, input_dim]
        输出:
            preds: [B] 回归预测值
        """
        x = x.transpose(1, 2)  # [B, input_dim, seq_len]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # [B, num_filters[-1]]
        x = self.dropout(x)
        out = self.fc(x).squeeze(-1)  # [B]
        return out
