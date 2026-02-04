import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import random
import pickle
from util import set_random_seed


def downsample_to_length(arr, target_length):
    """下采样到固定长度"""
    arr_tensor = torch.tensor(arr, dtype=torch.float32)
    arr_tensor = arr_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, L]
    pooled_tensor = F.adaptive_avg_pool1d(arr_tensor, output_size=target_length)
    downsampled_arr = pooled_tensor.squeeze(0).squeeze(0).numpy()
    return downsampled_arr


def process_and_save(name_list, set_path, downsample_seq_len, capacity):
    """处理全部样本并保存（不采样，直接用全量）"""
    all_samples = []

    for file_name in tqdm(name_list, desc=f"{capacity}Ah - 全量处理"):
        file_path = os.path.join(path, file_name)

        try:
            df = pd.read_csv(file_path)
        except Exception:
            continue

        try:
            # 获取放电容量
            max_capacity = df.query('工步号 == 3 and 工步名称 == "恒流放电"').iloc[-1]["容量 (mAh)"] / 1000

            # 获取恒流恒压充电段
            charge_df = df.query('工步号 == 1 and 工步名称 == "恒流恒压充电"')
            voltage = (charge_df["电压 (mV)"].values / 1000).tolist()  # mV -> V
            current = (charge_df["电流 (mA)"].values / 1000).tolist()  # mA -> A

            # 下采样
            voltage_curve = downsample_to_length(voltage, downsample_seq_len).tolist()
            current_curve = downsample_to_length(current, downsample_seq_len).tolist()

            sample = {
                "voltage_curve": voltage_curve,
                "current_curve": current_curve,
                "capacity": float(capacity),
                "max_capacity": float(max_capacity),
                "soh": float(max_capacity / capacity)
            }

            all_samples.append(sample)

        except Exception:
            continue

    # 保存
    with open(set_path, "wb") as f:
        pickle.dump(all_samples, f)


# ================= 主程序 =================

downsample_seq_len = 512
train_set_rate = 0.8
valid_set_rate = 0.1
test_set_rate = 0.1
assert abs(train_set_rate + valid_set_rate + test_set_rate - 1.0) < 1e-6

capacity_list = [15.5, 20, 23, 30, 52, 63, 67, 105]

# 42 418 621
random_seed = 621
set_random_seed(random_seed)

for capacity in tqdm(capacity_list, desc="容量循环"):
    path = f'./data/raw_data/{capacity}Ah/'
    output_dir = f'./data/generated_data_full_{random_seed}/'
    os.makedirs(output_dir, exist_ok=True)

    train_set_path = os.path.join(output_dir, f'{capacity}_train.pkl')
    valid_set_path = os.path.join(output_dir, f'{capacity}_valid.pkl')
    test_set_path = os.path.join(output_dir, f'{capacity}_test.pkl')

    # 删除旧文件
    if os.path.exists(train_set_path): os.remove(train_set_path)
    if os.path.exists(valid_set_path): os.remove(valid_set_path)
    if os.path.exists(test_set_path): os.remove(test_set_path)

    file_name_list = os.listdir(path)
    random.shuffle(file_name_list)
    file_len = len(file_name_list)

    train_name_list = file_name_list[: int(file_len * train_set_rate)]
    valid_name_list = file_name_list[int(file_len * train_set_rate): int(file_len * (train_set_rate + valid_set_rate))]
    test_name_list = file_name_list[int(file_len * (train_set_rate + valid_set_rate)):]

    # 全量处理并保存
    process_and_save(train_name_list, train_set_path, downsample_seq_len, capacity)
    process_and_save(valid_name_list, valid_set_path, downsample_seq_len, capacity)
    process_and_save(test_name_list, test_set_path, downsample_seq_len, capacity)

print(f"\n所有容量全量处理完成，数据已保存到 ./data/generated_data_full_{random_seed}/")
