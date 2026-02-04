# models/contrastive_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# ================== 连续型 SupConLoss ==================
class ContinuousSupConLoss(nn.Module):
    def __init__(self, temperature=0.1, sigma=0.1):
        super().__init__()
        self.temperature = temperature
        self.sigma = sigma

    def forward(self, features, labels):
        """
        features: [B, D] - 样本嵌入
        labels: [B] - 样本的 SOH（连续值）
        """
        device = features.device
        batch_size = features.size(0)

        # 标准化
        features = nn.functional.normalize(features, dim=1)

        # 相似度矩阵
        sim_matrix = torch.matmul(features, features.T) / self.temperature  # [B, B]

        # label相似度（高斯核）
        labels = labels.unsqueeze(1)  # [B, 1]
        label_diff = labels - labels.T
        label_sim = torch.exp(- (label_diff ** 2) / (2 * (self.sigma ** 2)))  # [B, B]

        # 去掉自身
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)

        # 正样本相似度
        exp_sim = torch.exp(sim_matrix) * mask
        pos_sim = exp_sim * label_sim

        # 损失
        loss = -torch.log((pos_sim.sum(dim=1) + 1e-8) / (exp_sim.sum(dim=1) + 1e-8))
        return loss.mean()
    

# # 二维矩阵先平均，再求 loss
# class ContinuousSupConLoss(nn.Module):
#     def __init__(self, temperature=0.1, sigma=0.1):
#         super().__init__()
#         self.temperature = temperature
#         self.sigma = sigma

#     def forward(self, features, labels):
#         """
#         features: [B, D, C] - 样本嵌入
#         labels: [B] - 样本的 SOH（连续值）
#         """
#         device = features.device
#         batch_size = features.size(0)

#         features = features.mean(dim=1)

#         # 标准化
#         features = nn.functional.normalize(features, dim=1)

#         # 相似度矩阵
#         sim_matrix = torch.matmul(features, features.T) / self.temperature  # [B, B]

#         # label相似度（高斯核）
#         labels = labels.unsqueeze(1)  # [B, 1]
#         label_diff = labels - labels.T
#         label_sim = torch.exp(- (label_diff ** 2) / (2 * (self.sigma ** 2)))  # [B, B]

#         # 去掉自身
#         mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)

#         # 正样本相似度
#         exp_sim = torch.exp(sim_matrix) * mask
#         pos_sim = exp_sim * label_sim

#         # 损失
#         loss = -torch.log((pos_sim.sum(dim=1) + 1e-8) / (exp_sim.sum(dim=1) + 1e-8))
#         return loss.mean()
    

# 二维矩阵求 loss
# class ContinuousSupConLoss(nn.Module):
#     def __init__(self, temperature=0.1, sigma=0.1, sim_type="bilinear", D=None, C=None):
#         """
#         features: [B, D, C] - 样本嵌入
#         labels: [B] - 样本的 SOH（连续值）

#         Args:
#             temperature: softmax 温度
#             sigma: 标签相似度的高斯核宽度
#             sim_type: 'frobenius' 或 'bilinear'
#             D, C: 当 sim_type='bilinear' 时需要输入矩阵大小，用于定义 W
#         """
#         super().__init__()
#         self.temperature = temperature
#         self.sigma = sigma
#         self.sim_type = sim_type

#         if sim_type == "bilinear":
#             assert D is not None, "Bilinear 需要 D"
#             # W 可以设计在 D 维或 C 维，这里默认放在 D 上
#             self.W = nn.Parameter(torch.randn(D, D))

#     def compute_similarity(self, A, B):
#         """
#         A, B: [B, D, C]
#         return: sim_matrix [B, B]
#         """
#         if self.sim_type == "frobenius":
#             # flatten 到 [B, D*C] 再做点积（等价于 Frobenius inner product）
#             A_flat = A.reshape(A.size(0), -1)  # [B, D*C]
#             B_flat = B.reshape(B.size(0), -1)
#             A_norm = F.normalize(A_flat, dim=1)
#             B_norm = F.normalize(B_flat, dim=1)
#             sim = torch.matmul(A_norm, B_norm.T)  # [B, B]
#         elif self.sim_type == "bilinear":
#             # bilinear: Tr(A^T W B)
#             # => sum_c (A[:,:,c]^T W B[:,:,c])
#             B_size, D, C = A.shape
#             sim = torch.zeros(B_size, B_size, device=A.device)
#             self.W.to(A.device)
#             for c in range(C):
#                 A_c = A[:, :, c]  # [B, D]
#                 B_c = B[:, :, c]  # [B, D]
#                 sim += torch.matmul(torch.matmul(A_c, self.W), B_c.T)  # [B, B]
#             # 归一化
#             sim = sim / C
#             sim = sim / self.temperature
#         else:
#             raise ValueError(f"未知的 sim_type: {self.sim_type}")
#         return sim

#     def forward(self, features, labels):
#         """
#         features: [B, D, C]
#         labels: [B] (连续值)
#         """
#         device = features.device
#         B = features.size(0)

#         # 相似度矩阵 [B, B]
#         sim_matrix = self.compute_similarity(features, features)

#         # 标签相似度（高斯核）
#         labels = labels.unsqueeze(1)  # [B, 1]
#         label_diff = labels - labels.T
#         label_sim = torch.exp(- (label_diff ** 2) / (2 * (self.sigma ** 2)))  # [B, B]

#         # 去掉自身
#         mask = ~torch.eye(B, dtype=torch.bool, device=device)

#         # exp(sim) 并 masked
#         exp_sim = torch.exp(sim_matrix) * mask
#         pos_sim = exp_sim * label_sim

#         # InfoNCE loss
#         loss = -torch.log((pos_sim.sum(dim=1) + 1e-8) / (exp_sim.sum(dim=1) + 1e-8))
#         return loss.mean()