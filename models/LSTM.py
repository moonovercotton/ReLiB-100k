import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, num_layers=2):
        """
        input_dim: 每个时间步的特征维度，这里是 2
        hidden_dim: LSTM 隐状态维度
        num_layers: LSTM 层数
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM 层
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # 回归头
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        输入:
            x: [B, seq_len, 2]
        输出:
            preds: [B]
        """
        B, seq_len, _ = x.shape
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, B, self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, B, self.hidden_dim, device=x.device)

        # LSTM 前向传播
        out, (hn, cn) = self.lstm(x, (h0, c0))  # out: [B, seq_len, hidden_dim]

        # 取最后一个时间步的输出
        last_out = out[:, -1, :]  # [B, hidden_dim]

        # 回归预测
        preds = self.fc(last_out).squeeze(-1)  # [B]
        return preds
