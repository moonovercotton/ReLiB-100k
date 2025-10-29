import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=128):
        super().__init__()
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
            x: [B, seq_len, 2] 数据集原始形状
        输出:
            preds: [B] 回归预测值（标量）
        """
        B, seq_len, v_num = x.shape
        out = self.model(x.reshape(B, seq_len * v_num)).squeeze(-1)  # squeeze成[B]
        return out
