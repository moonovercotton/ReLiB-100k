import os
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.multiprocessing as mp
import sys

from dataset import BatteryDataset
from models.Ours import CNNEncoder, TCNEncoder, RNNEncoder, LinearEncoder
from models.contrastive_loss import ContinuousSupConLoss

# ================== 训练函数 ==================
def train_contrastive(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for view1, view2, soh, _ in dataloader:
        view1, view2, soh = view1.to(device), view2.to(device), soh.to(device)

        # 拼接视图
        inputs = torch.cat([view1, view2], dim=0)   # [2B, T, C]
        embeddings = model(inputs)                  # [2B, D]

        labels = soh.repeat(2)  # [2B]

        loss = criterion(embeddings, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# ================== 单个容量训练函数 ==================
def run_on_gpu(gpu_id, cap, train_type, model_name):
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"Ours_pretrain_{cap}.txt")
    sys.stdout = open(output_path, "w", buffering=1)
    
    if train_type == 'single':
        data_file = f"./data/generated_data/{cap}_train" 
        model_save_path = f"./cache/saved_models/{cap}_contrastive_encoder.pth" 
        loss_plot_path = f'./plot/loss_plot/{model_name}_{cap}Ah_contrastive_loss.png' 
    elif train_type == 'mix': 
        data_file = f"./data/generated_data/multi_without_{cap}_train" 
        model_save_path = f"./cache/saved_models/multi_without_{cap}_contrastive_encoder.pth" 
        loss_plot_path = f'./plot/loss_plot/{model_name}_multi_without_{cap}Ah_contrastive_loss.png'


    batch_size = 64
    epochs_pretrain = 300
    lr = 1e-3
    out_dim = 64

    # ===== 数据加载 =====
    dataset = BatteryDataset(data_file, flag='contrastive')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 获取输入长度
    sample_feature, _, _, _ = dataset[0]
    seq_len = sample_feature.shape[0]
    # print(f'输入长度: {seq_len}')
    # exit(0)

    # ===== 模型与优化器 =====
    encoder = CNNEncoder(input_dim=2, hidden_dim=1024, out_dim=out_dim).to(device)
    # encoder = TCNEncoder(input_dim=2, out_dim=out_dim).to(device)
    # encoder = RNNEncoder(input_dim=2, hidden_dim=256, out_dim=out_dim).to(device)
    # encoder = LinearEncoder(input_dim=2, seq_len=512, out_dim=out_dim).to(device)

    contrastive_loss = ContinuousSupConLoss(temperature=0.07, sigma=0.1)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)

    print(f"[GPU {gpu_id}] [{train_type}] Start training  {cap}Ah ...")
    train_loss_list = []

    # ===== 早停参数 =====
    patience = 20
    min_delta = 1e-4
    best_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    wait = 0

    for epoch in tqdm(range(epochs_pretrain), desc=f'Ours_{cap}', file=sys.stdout):
        loss = train_contrastive(encoder, dataloader, contrastive_loss, optimizer, device)
        train_loss_list.append(loss)
        print(f"\n[GPU {gpu_id}] [Epoch {epoch+1}] Loss: {loss:.4f}")

        # ===== 早停逻辑 =====
        if best_loss - loss > min_delta:
            best_loss = loss
            best_epoch = epoch
            best_model_state = encoder.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"\n[GPU {gpu_id}] Early stopping at epoch {epoch+1}, best epoch was {best_epoch+1}")
                break

    # 恢复最佳模型
    if best_model_state is not None:
        encoder.load_state_dict(best_model_state)

    # ===== 绘制损失曲线 =====
    os.makedirs("./plot/loss_plot", exist_ok=True)
    plt.figure(figsize=(8, 5))
    epochs_range = range(1, len(train_loss_list)+1)
    plt.plot(epochs_range, train_loss_list, label='Train Loss')
    best_epoch_plot = best_epoch + 1
    plt.axvline(x=best_epoch_plot, color='r', linestyle='--', label='Best Epoch')
    plt.scatter(best_epoch_plot, train_loss_list[best_epoch], color='r')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Contrastive Pretraining Loss ({cap}Ah)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(loss_plot_path)
    plt.close()

    # ===== 保存模型 =====
    os.makedirs("./cache/saved_models", exist_ok=True)
    torch.save(encoder.state_dict(), model_save_path)
    print(f"[GPU {gpu_id}] Encoder model saved to {model_save_path}")


# ================== 主程序 ==================
def main():
    CAPACITY_LIST = ['15.5', '20', '23', '30', '52', '63', '67', '105']
    # CAPACITY_LIST = ['20', '30']

    train_type = 'single'
    model_name = "CNNEncoder"
    # model_name = "TCNEncoder"
    # model_name = "RNNEncoder"
    # model_name = "LinearEncoder"

    # available_gpus = [0, 1, 2, 3, 4, 5, 6, 7]
    available_gpus = [0, 1, 2, 3]  # 只使用 GPU 2 和 3
    batch_size = len(available_gpus)

    # 将容量列表按 batch_size 切分
    for i in range(0, len(CAPACITY_LIST), batch_size):
        batch = CAPACITY_LIST[i:i+batch_size]
        processes = []
        for gpu_id, cap in zip(available_gpus, batch):
            p = mp.Process(target=run_on_gpu, args=(gpu_id, cap, train_type, model_name))
            p.start()
            processes.append(p)

        # 等待本批次完成
        for p in processes:
            p.join()



if __name__ == "__main__":
    mp.set_start_method('spawn')  # 在 Windows 或多 GPU 上推荐使用 spawn
    main()
