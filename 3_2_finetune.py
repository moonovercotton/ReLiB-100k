# 3_1_train_regression.py
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
# 从外部引入模块
from models.Ours import regression_head
from dataset import BatteryDataset
from util import count_parameters
from models.Ours import CNNEncoder, TCNEncoder, RNNEncoder, LinearEncoder


# ========== 训练函数 ==========
def train_regression(encoder, regressor, dataloader, criterion, optimizer, device):
    # encoder.eval()  # 冻结 encoder
    encoder.train()
    regressor.train()
    total_loss = 0
    for features, labels, _ in dataloader:
        features = features.to(device)
        labels = labels.to(device)

        # with torch.no_grad():  # encoder 不更新
        #     embeddings = encoder(features)
        embeddings = encoder(features)
        # w/o CNNEncoder
        # embeddings = features.reshape(features.shape[0], -1)

        preds = regressor(embeddings).squeeze()
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate_regression(encoder, regressor, dataloader, criterion, device):
    encoder.eval()
    regressor.eval()
    total_loss = 0
    preds_all, labels_all = [], []

    with torch.no_grad():
        for features, labels, _ in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            embeddings = encoder(features)
            # w/o CNNEncoder
            # embeddings = features.reshape(features.shape[0], -1)
            preds = regressor(embeddings).squeeze()

            loss = criterion(preds, labels)
            total_loss += loss.item()

            preds_all.append(preds.cpu().numpy())
            labels_all.append(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    preds_all = np.concatenate(preds_all)
    labels_all = np.concatenate(labels_all)

    rmse = np.sqrt(np.mean((preds_all - labels_all) ** 2))
    mae = np.mean(np.abs(preds_all - labels_all))
    mape = np.mean(np.abs((preds_all - labels_all) / labels_all)) * 100

    return avg_loss, rmse, mae, mape


# ========== 主程序 ==========
def main(capacity=15.5,
         flag='benchmark',
         epochs_regression=300,
         enc_out_dim=2048,
         batch_size=64,
         lr=1e-3, 
         dropout=0.1, 
         embedding_dim=512,
         device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
         ):
    
    
    # ======== 加载数据 =========
    if flag == 'benchmark':
        # model_save_path = f'./cache/saved_models/{capacity}_contrastive_encoder.pth'
        model_save_path = f'./cache/saved_models/20_contrastive_encoder.pth'
        train_set = BatteryDataset(f"./data/generated_data/{capacity}_train", flag='cap_regression')
        valid_set = BatteryDataset(f"./data/generated_data/{capacity}_valid", flag='cap_regression')
        test_set = BatteryDataset(f"./data/generated_data/{capacity}_test", flag='cap_regression')
        loss_path = f"./plot/loss_plot/Ours_benchmark_{capacity}.png"
    elif flag == 'zero_shot':
        model_save_path = f'./cache/saved_models/multi_without_{capacity}_contrastive_encoder.pth'
        train_set = BatteryDataset(f"./data/generated_data/multi_without_{capacity}_train", flag='soh_regression')
        valid_set = BatteryDataset(f"./data/generated_data/multi_without_{capacity}_valid", flag='soh_regression')
        test_set = BatteryDataset(f"./data/generated_data/{capacity}_test", flag='soh_regression')
        loss_path = f"./plot/loss_plot/Ours_zero_shot_{capacity}.png"


    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # 获取输入长度
    sample_feature, _, _ = valid_set[0]
    seq_len = sample_feature.shape[0]
    # print(f'输入长度: {seq_len}')
    # exit(0)

    # ======== 加载 encoder =========
    encoder = CNNEncoder(input_dim=2, hidden_dim=1024, out_dim=enc_out_dim).to(device)
    # encoder = TCNEncoder(input_dim=2, out_dim=enc_out_dim).to(device)
    # encoder = RNNEncoder(input_dim=2, hidden_dim=256, out_dim=enc_out_dim).to(device)
    # encoder = LinearEncoder(input_dim=2, seq_len=512, out_dim=enc_out_dim).to(device)

    encoder.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))
    # encoder.eval()
    print(f"Loaded pretrained encoder from {model_save_path}")

    # ======== 初始化回归头 =========
    # MLP
    regressor = regression_head(input_dim=enc_out_dim, hidden_dim=embedding_dim).to(device)
    # w/o CNNEncoder
    # regressor = regression_head(input_dim=1024, hidden_dim=embedding_dim).to(device)
    optimizer = torch.optim.Adam(regressor.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()


    encoder_params = count_parameters(encoder)
    print(f"Encoder 可训练参数量: {encoder_params:,}")
    regressor_params = count_parameters(regressor)
    print(f"Regressor 可训练参数量: {regressor_params:,}")
    total_params = encoder_params + regressor_params
    print(f"总可训练参数量: {total_params:,}")
    exit(0)

    # ========= 训练回归模型 =========
    print("Start regression training...")
    train_loss_list, valid_loss_list = [], []

    # ===== 早停参数 =====
    patience = 20        # 连续多少轮验证 loss 没有下降就停止
    min_delta = 1e-4     # 最小改善幅度
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    wait = 0

    for epoch in tqdm(range(epochs_regression), desc=f'Ours_{capacity}', file=sys.stdout):
        train_loss = train_regression(encoder, regressor, train_loader, criterion, optimizer, device)
        valid_loss, valid_rmse, valid_mae, valid_mape = evaluate_regression(encoder, regressor, valid_loader, criterion, device)

        print(f"\n[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | "
            f"Valid Loss: {valid_loss:.4f} | RMSE: {valid_rmse:.4f} | "
            f"MAE: {valid_mae:.4f} | MAPE: {valid_mape:.2f}%")

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

        # ===== 早停逻辑 =====
        if best_val_loss - valid_loss > min_delta:
            best_val_loss = valid_loss
            best_epoch = epoch
            best_model_state = regressor.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1}, best epoch was {best_epoch+1}")
                break

    # 恢复最佳模型
    if best_model_state is not None:
        regressor.load_state_dict(best_model_state)

    # ========== 绘制损失曲线 ==========
    os.makedirs("./plot/loss_plot", exist_ok=True)
    plt.figure(figsize=(8, 5))
    epochs_range = range(1, len(train_loss_list)+1)
    plt.plot(epochs_range, train_loss_list, label="Train Loss")
    plt.plot(epochs_range, valid_loss_list, label="Valid Loss")

    # 标记最佳验证点
    best_epoch_plot = best_epoch + 1
    plt.axvline(x=best_epoch_plot, color='r', linestyle='--', label='Best Epoch')
    plt.scatter(best_epoch_plot, valid_loss_list[best_epoch], color='r')

    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title(f"{capacity} Ah {flag} finetune Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(loss_path)
    plt.close()

    # ======= 测试集评估 =======
    test_loss, test_rmse, test_mae, test_mape = evaluate_regression(encoder, regressor, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | RMSE: {test_rmse:.4f} | "
          f"MAE: {test_mae:.4f} | MAPE: {test_mape:.2f}%")
    print(f'RMSE MAE MAPE:\n{test_rmse:.3f} {test_mae:.3f} {test_mape:.2f}%')


if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser()
    parser.add_argument('--capacity', type=float, help="电池额定容量")
    parser.add_argument('--flag', type=str, default="benchmark")
    parser.add_argument('--epochs_regression', type=int, default=300)
    parser.add_argument('--enc_out_dim', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embedding_dim', type=int, default=512)
    parser.add_argument("--gpu_id", type=int, default=0)

    # 只在命令行有额外参数时才解析
    if len(sys.argv) > 1:
        args = parser.parse_args()
        main(
            capacity=int(args.capacity) if args.capacity.is_integer() else args.capacity,
            flag=args.flag,
            epochs_regression=args.epochs_regression,
            enc_out_dim=args.enc_out_dim,
            batch_size=args.batch_size,
            lr=args.lr,
            dropout=args.dropout,
            embedding_dim=args.embedding_dim,
            device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
        )
    else:
        # 没有命令行参数时，使用默认调用
        main(capacity=23, flag='benchmark', epochs_regression=300, enc_out_dim=64)