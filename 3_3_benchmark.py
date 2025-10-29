import os
import torch
from torch.utils.data import DataLoader
from dataset import BatteryDataset
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from util import count_parameters
import sys

# 从外部引入模块
from models import MLP
from models import iTransformer, TimesNet, DLinear
from models import SparseTSF, PatchTST
from models import LSTM, TCN, CNN, TimeMixer


def train_regression(regressor, dataloader, criterion, optimizer, device, capacity):

    regressor.train()
    total_loss = 0
    for features, labels, _ in dataloader:
        # print(features[0])
        # print(labels[0])
        # exit(0)

        features = features.to(device)
        labels = labels.to(device)

        # with torch.no_grad():
        #     embeddings = encoder(features)  # [batch, seq_len, encoder_dim]

        preds = regressor(features).squeeze()  # [batch]
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate_regression(regressor, dataloader, criterion, device):
    regressor.eval()
    total_loss = 0
    preds_all = []
    labels_all = []

    with torch.no_grad():
        for features, labels, _ in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            preds = regressor(features).squeeze()

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


def main(capacity, model_name="MLP", epochs_regression=300, batch_size=64, lr=1e-4, dropout=0.1, embedding_dim=512,
         device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
         ):
    print(f"Running: capacity={capacity}, model={model_name}, "
          f"epochs={epochs_regression}, batch_size={batch_size}, "
          f"lr={lr}, dropout={dropout}, emb_dim={embedding_dim}")
    

    # ======== 加载数据 =========
    train_set = BatteryDataset(f'./data/generated_data/{capacity}_train', flag='cap_regression')
    valid_set = BatteryDataset(f'./data/generated_data/{capacity}_valid', flag='cap_regression')
    test_set = BatteryDataset(f'./data/generated_data/{capacity}_test', flag='cap_regression')

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # 获取输入长度
    sample_feature, _, _ = valid_set[0]
    seq_len = sample_feature.shape[0]
    # print(f'输入长度: {seq_len}')
    # exit(0)

    # ========= 回归训练 =========
    if model_name == "MLP":
        regressor = MLP(input_dim=seq_len*2).to(device)
        regression_optimizer = torch.optim.Adam(regressor.parameters(), lr=lr)
        regression_loss = torch.nn.MSELoss()
    elif model_name == "LSTM":
        hidden_dim = embedding_dim   # 可以根据需要调整
        num_layers = 2     # 可以根据需要调整
        regressor = LSTM(input_dim=2, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
        regression_optimizer = torch.optim.Adam(regressor.parameters(), lr=lr)
        regression_loss = torch.nn.MSELoss()
    elif model_name == "TCN":
        num_channels = [128, 128, 128]  # 可以根据需要调整每层通道数
        kernel_size = 3
        regressor = TCN(input_dim=2, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout).to(device)
        regression_optimizer = torch.optim.Adam(regressor.parameters(), lr=lr)
        regression_loss = torch.nn.MSELoss()
    elif model_name == "CNN":
        regressor = CNN(input_dim=2, seq_len=seq_len).to(device)
        regression_optimizer = torch.optim.Adam(regressor.parameters(), lr=lr)
        regression_loss = torch.nn.MSELoss()
    elif model_name == "iTransformer": 
        class iTransformerRegressionConfig:
            def __init__(self):
                self.seq_len = seq_len
                self.pred_len = 1
                self.d_model = embedding_dim
                self.n_heads = 8    # default
                self.e_layers = 3
                self.d_ff = 512
                self.dropout = dropout
                self.output_attention = False    # default
                self.embed = 'timeF'    # default
                self.freq = 's'
                self.factor = 1    # default
                self.use_norm = True    # default
                self.class_strategy = 'projection'    # default
                self.activation = 'gelu'    # default

                # extra
                self.c_out = 1

        config = iTransformerRegressionConfig()
        regressor = iTransformer(config).to(device)
        regression_optimizer = torch.optim.Adam(regressor.parameters(), lr=lr)
        regression_loss = torch.nn.MSELoss()
    elif model_name == "TimesNet":
        class TimesNetRegressionConfig:
            def __init__(self):
                self.task_name = 'short_term_forecast'
                self.seq_len = seq_len
                self.pred_len = 1
                self.label_len = 1
                self.d_model = embedding_dim
                self.enc_in = 2    # input channels
                self.c_out = 1    # output channels
                # self.n_heads = 8    # default
                self.e_layers = 2
                self.d_ff = 32
                self.top_k = 3
                self.dropout = dropout    # default
                # self.output_attention = False    # default
                self.embed = 'timeF'    # default
                self.freq = 's'  
                # self.factor = 1    # default
                # self.use_norm = True    # default
                # self.class_strategy = 'projection'    # default
                # self.activation = 'gelu'    # default
                self.num_kernels = 6    # default

        config = TimesNetRegressionConfig()
        regressor = TimesNet(config).to(device)
        regression_optimizer = torch.optim.Adam(regressor.parameters(), lr=lr)
        regression_loss = torch.nn.MSELoss()
    elif model_name == "TimeMixer":
        class TimeMixerRegressionConfig:
            def __init__(self):
                self.task_name = 'short_term_forecast'
                self.seq_len = seq_len
                self.pred_len = 1
                self.label_len = 1
                self.down_sampling_window = 2
                self.channel_independence = 0
                self.d_model = embedding_dim
                self.moving_avg = 25
                self.enc_in = 2    # input channels
                self.c_out = 1    # output channels
                self.use_future_temporal_feature = False
                # self.n_heads = 8    # default
                self.e_layers = 4
                self.d_ff = 32
                self.top_k = 3
                self.dropout = dropout    # default
                # self.output_attention = False    # default
                self.embed = 'timeF'    # default
                self.freq = 's'  
                # self.factor = 1    # default
                self.use_norm = True    # default
                # self.class_strategy = 'projection'    # default
                # self.activation = 'gelu'    # default
                self.num_kernels = 6    # default
                self.down_sampling_layers = 1
                self.decomp_method = 'moving_avg'
                self.down_sampling_method = 'avg'

                self.c_out = 1

        config = TimeMixerRegressionConfig()
        regressor = TimeMixer(config).to(device)
        regression_optimizer = torch.optim.Adam(regressor.parameters(), lr=lr)
        regression_loss = torch.nn.MSELoss()
    elif model_name == "DLinear":
        class DLinearRegressionConfig:
            def __init__(self):
                self.task_name = 'long_term_forecast'
                self.seq_len = seq_len
                self.pred_len = 1
                self.enc_in = 2   # input channels
                self.moving_avg = 25    # 移动平均窗口大小（传递给 series_decomp）; default

                self.c_out = 1   # output channels

        config = DLinearRegressionConfig()
        regressor = DLinear(config).to(device)
        regression_optimizer = torch.optim.Adam(regressor.parameters(), lr=lr)
        regression_loss = torch.nn.MSELoss()
    elif model_name == "SparseTSF":
        class SparseTSFRegressionConfig:
            def __init__(self):
                self.seq_len = seq_len
                self.pred_len = 1
                self.enc_in = 2
                self.period_len = 1
                self.d_model = embedding_dim
                self.model_type = 'mlp'

        config = SparseTSFRegressionConfig()
        sparsetsf_model = SparseTSF(config).to(device)
        regressor = sparsetsf_model
        regression_optimizer = torch.optim.Adam(sparsetsf_model.parameters(), lr=lr)
        regression_loss = torch.nn.MSELoss()
    elif model_name == "PatchTST":
        class PatchTSTRegressionConfig:
            def __init__(self):
                self.seq_len = seq_len
                self.pred_len = 1
                self.enc_in = 2
                self.c_out = 1
                self.d_model = embedding_dim
                self.n_heads = 8
                self.e_layers = 2
                self.d_ff = 128
                self.dropout = dropout
                self.activation = 'gelu'
                self.patch_len = 16
                self.stride = 8
                self.max_seq_len = 1024
                self.fc_dropout = 0.1
                self.head_dropout = 0.1
                self.individual = False
                self.padding_patch = 'end'
                self.revin = True
                self.affine = True
                self.subtract_last = False
                self.decomposition = False
                self.kernel_size = 25
                self.norm = 'BatchNorm'
                self.attn_dropout = 0.1
                self.key_padding_mask = 'auto'
                self.padding_var = None
                self.attn_mask = None
                self.res_attention = True
                self.pre_norm = False
                self.store_attn = False
                self.pe = 'zeros'
                self.learn_pe = True
                self.pretrain_head = False
                self.head_type = 'flatten'
                self.verbose = False

        config = PatchTSTRegressionConfig()
        patchtst_model = PatchTST(config).to(device)
        regressor = patchtst_model
        regression_optimizer = torch.optim.Adam(patchtst_model.parameters(), lr=lr)
        regression_loss = torch.nn.MSELoss()

    # # 记录模型参数量
    # num_params = sum(p.numel() for p in regressor.parameters())
    # print(f'param nums: {num_params:,}')
    # exit(0)
    # with open("./models_parameters.txt", "a", encoding="utf-8") as f:
    #     f.write(f"{model_name}: {num_params}\n")

    print("Start regression...")
    train_loss_list = list()
    valid_loss_list = list()

    # regressor_params = count_parameters(regressor)
    # print(f"Regressor 可训练参数量: {regressor_params:,}")
    # exit(0)

    # ========== 早停参数 ==========
    patience = 20  # 可以根据需求调整
    best_val_loss = float("inf")
    best_epoch = 0
    best_model_state = None
    wait = 0

    for epoch in tqdm(range(epochs_regression), desc=f"{model_name}_{capacity}", file=sys.stdout):
        train_loss = train_regression(regressor, train_loader, regression_loss, regression_optimizer, device, capacity)
        valid_loss, valid_rmse, valid_mae, valid_mape = evaluate_regression(regressor, valid_loader, regression_loss, device)
        print(f'\n[Regression Epoch {epoch + 1}] Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f} | Valid RMSE: {valid_rmse:.4f} | Valid MAE: {valid_mae:.4f} | Valid MAPE: {valid_mape:.2f}%')

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

        # ====== 早停逻辑 ======
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            best_epoch = epoch
            best_model_state = regressor.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}, best epoch was {best_epoch+1}")
                break

    # 恢复最佳模型
    if best_model_state is not None:
        regressor.load_state_dict(best_model_state)

    # ========== 绘制损失曲线 ==========
    plt.figure(figsize=(8, 5))
    epochs_range = range(1, len(train_loss_list) + 1)  # 横坐标用实际训练轮数
    plt.plot(epochs_range, train_loss_list, label='Train Loss')
    plt.plot(epochs_range, valid_loss_list, label='Valid Loss')
    # 标记最佳验证点
    best_epoch_plot = best_epoch + 1  # Python索引从0开始
    plt.axvline(x=best_epoch_plot, color='r', linestyle='--', label='Best Epoch')
    plt.scatter(best_epoch_plot, valid_loss_list[best_epoch], color='r')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'./plot/loss_plot/{model_name}_benchmark_{capacity}.png')
    plt.close()

    # ======= 测试集评估 =======
    test_loss, test_rmse, test_mae, test_mape = evaluate_regression(regressor, test_loader, regression_loss, device)
    print(f'\nTest Loss: {test_loss:.4f} | Test RMSE: {test_rmse:.4f} | Test MAE: {test_mae:.4f} | Test MAPE: {test_mape:.2f}%')
    print(f'{capacity} {model_name}')
    print(f'RMSE MAE MAPE:\n{test_rmse:.3f} {test_mae:.3f} {test_mape:.2f}%')


if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser()
    parser.add_argument('--capacity', type=float, help="电池额定容量")
    parser.add_argument('--model_name', type=str, default="MLP")
    parser.add_argument('--epochs_regression', type=int, default=300)
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
            model_name=args.model_name,
            epochs_regression=args.epochs_regression,
            batch_size=args.batch_size,
            lr=args.lr,
            dropout=args.dropout,
            embedding_dim=args.embedding_dim,
            device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
        )
    else:
        # 没有命令行参数时，使用默认调用
        main(capacity=15.5, model_name="LSTM", epochs_regression=300)