import torch
import torch.nn as nn
import torch.nn.functional as F

class Chomp1d(nn.Module):
    """去掉卷积后多余的时间步"""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()  # 保证连续内存

class TemporalBlock(nn.Module):
    """TCN的基本残差块"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_dim=2, num_channels=[128, 128, 128], kernel_size=3, dropout=0.2):
        """
        input_dim: 输入特征维度（v_num）
        num_channels: 每个TCN层的输出通道数
        """
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            in_ch = input_dim if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            dilation_size = 2 ** i
            padding = (kernel_size - 1) * dilation_size
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, stride=1, dilation=dilation_size, padding=padding, dropout=dropout))
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)  # 输出回归标量

    def forward(self, x):
        """
        输入:
            x: [B, seq_len, v_num]
        输出:
            preds: [B] 回归预测值
        """
        x = x.transpose(1, 2)  # 转成 [B, v_num, seq_len] 适配Conv1d
        y = self.tcn(x)
        y = y[:, :, -1]  # 取最后一个时间步的输出
        return self.fc(y).squeeze(-1)
