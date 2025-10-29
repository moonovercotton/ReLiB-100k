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
    arr_tensor = arr_tensor.unsqueeze(0).unsqueeze(0)
    pooled_tensor = F.adaptive_avg_pool1d(arr_tensor, output_size=target_length)
    downsampled_arr = pooled_tensor.squeeze(0).squeeze(0).numpy()
    return downsampled_arr


def random_sampling(name_list, set_path, sampling_num, downsample_seq_len, capacity):
    """随机采样（均匀随机抽样）"""
    all_samples = []

    # 如果文件数少于需要的采样数量，就随机重复采样
    if len(name_list) >= sampling_num:
        selected_files = random.sample(name_list, sampling_num)
    else:
        print(f'文件数量少于需要采样的数量，随机重复采样')
        repeat_times = sampling_num // len(name_list)
        remain = sampling_num % len(name_list)
        selected_files = name_list * repeat_times + random.sample(name_list, remain)

    for file_name in tqdm(selected_files, desc=f"{capacity}Ah - 随机采样"):
        try:
            df = pd.read_csv(os.path.join(path, file_name))
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

    # 保存采样结果
    with open(set_path, 'wb') as f:
        pickle.dump(all_samples, f)


# ================= 主程序 =================

sampling_num = 3000
downsample_seq_len = 512
train_set_rate = 0.8
valid_set_rate = 0.1
test_set_rate = 0.1
assert abs(train_set_rate + valid_set_rate + test_set_rate - 1.0) < 1e-6

capacity_list = [15.5, 20, 23, 30, 52, 63, 67, 105]

set_random_seed()

for capacity in tqdm(capacity_list, desc="容量循环"):
    path = f'./data/raw_data/{capacity}Ah/'
    output_dir = './data/generated_data/'
    os.makedirs(output_dir, exist_ok=True)

    train_set_path = os.path.join(output_dir, f'{capacity}_train')
    valid_set_path = os.path.join(output_dir, f'{capacity}_valid')
    test_set_path = os.path.join(output_dir, f'{capacity}_test')

    # 删除旧文件
    if os.path.exists(train_set_path): os.remove(train_set_path)
    if os.path.exists(valid_set_path): os.remove(valid_set_path)
    if os.path.exists(test_set_path): os.remove(test_set_path)

    file_name_list = os.listdir(path)
    random.shuffle(file_name_list)
    file_len = len(file_name_list)

    train_name_list = file_name_list[0: int(file_len * train_set_rate)]
    valid_name_list = file_name_list[int(file_len * train_set_rate): int(file_len * (train_set_rate + valid_set_rate))]
    test_name_list = file_name_list[int(file_len * (train_set_rate + valid_set_rate)):]

    # 分别随机采样并保存
    random_sampling(train_name_list, train_set_path, int(sampling_num * train_set_rate), downsample_seq_len, capacity)
    random_sampling(valid_name_list, valid_set_path, int(sampling_num * valid_set_rate), downsample_seq_len, capacity)
    random_sampling(test_name_list, test_set_path, int(sampling_num * test_set_rate), downsample_seq_len, capacity)

print("\n所有容量随机采样完成，数据已保存到 ./data/generated_data/")
