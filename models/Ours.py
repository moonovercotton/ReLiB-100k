import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision
from layers.Embed import DataEmbedding

class regression_head(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128):
        super(regression_head, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        输入:
            x: [B, 64] 编码器输出的特征向量
        输出:
            preds: [B] 回归预测值（标量）
        """
        out = self.model(x).squeeze(-1)  # squeeze成[B]
        return out


class CNNEncoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, out_dim=64):
        super(CNNEncoder, self).__init__()
        self.seq_len = 512
        self.pred_len = 1

        self.enc_embedding = DataEmbedding(c_in=input_dim, d_model=hidden_dim, embed_type='fixed', freq='s',
                                    dropout=0.1)
        self.conv1 = nn.Conv1d(in_channels=hidden_dim, out_channels=128, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        # Normalization from Non-stationary Transformer
        means = x.mean(1, keepdim=True).detach()
        x = x.sub(means)
        stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x.div(stdev)

        # 输入: [B, T, 2]
        x = self.enc_embedding(x, None)
        x = x.permute(0, 2, 1)  # => [B, 2, T]
        x = F.relu(self.bn1(self.conv1(x))) # => [B, hidden_dim, T]
        x = F.relu(self.bn2(self.conv2(x))) # => [B, hidden_dim, T]
        x = self.pool(x)  # => [B, hidden_dim, 1]
        x = x.squeeze(-1)  # => [B, hidden_dim]
        x = self.fc(x)   # => [B, out_dim]
        x = F.normalize(x, dim=1)

        return x
    

# ========== 简单线性编码器（线性回归用） ==========
class LinearEncoder(nn.Module):
    def __init__(self, input_dim=2, seq_len=512, out_dim=64):
        super(LinearEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),  # [B, T, 2] => [B, T*2]
            nn.Linear(seq_len * input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_dim)
        )

    def forward(self, x):
        return self.fc(x)  # => [B, out_dim]
    

from pytorch_tcn import TCN
import torch.nn as nn
import torch


# ========== TCN 编码器 ==========
class TCNEncoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dims=[256, 1024, 256], out_dim=64, kernel_size=3, dropout=0.2):
        super(TCNEncoder, self).__init__()
        self.tcn = TCN(
            num_inputs=input_dim,
            num_channels=hidden_dims,
            kernel_size=kernel_size,
            dropout=dropout,
            causal=True,
            input_shape='NLC'  # 支持 [batch, length, channels]
        )
        self.fc = nn.Linear(hidden_dims[-1], out_dim)

    def forward(self, x):
        # x: [B, T, C]
        out = self.tcn(x)         # 输出 shape：[B, T, hidden_dims[-1]]
        out = out.mean(dim=1)     # 全局池化（取均值） → [B, hidden_dims[-1]]
        out = self.fc(out)        # → [B, out_dim]
        return nn.functional.normalize(out, dim=1)


# ========== RNN 编码器 ==========
class RNNEncoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, out_dim=64, num_layers=2, bidirectional=True):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, 
                           batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), out_dim)
        
    def forward(self, x):
        # x: [B, T, C]
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # 取最后一个时间步
        out = self.fc(out)
        return F.normalize(out, dim=1)
