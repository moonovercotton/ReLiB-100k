import itertools
import numpy as np
import subprocess
from tqdm import tqdm
import json
import os
import torch
import torch.multiprocessing as mp

# =========================
# 配置
# =========================
# model_names = ["MLP", "LSTM", "TCN", "CNN", "iTransformer", "TimesNet", "DLinear", "PatchTST", "SparseTSF",
#                "TimeMixer"]  # 可扩展
model_names = ["PatchTST"]
capacities = [15.5, 20, 23, 30, 52, 63, 67, 105]
# capacities = [52, 63, 67, 105]
# capacities = [15.5]

use_gpus = True         # True: 每个容量一个 GPU; False: 在单个 GPU 上串行运行
single_gpu_id = 3       # use_gpus=False 时指定 GPU 编号


batch_sizes = [16, 32, 64, 128]
learning_rates = [1e-4, 5e-4, 1e-3, 5e-3]
dropouts = [0, 0.05, 0.1, 0.2]
embedding_dims = [32, 64, 128, 256, 512]

default_params = {
    "batch_size": 64,
    "lr": 1e-4,
    "dropout": 0.1,
    "embedding_dim": 512,
}

search_capacity = capacities[0]

os.makedirs("./cache/best_params", exist_ok=True)
os.makedirs("./outputs", exist_ok=True)

# =========================
# 评估函数
# =========================
def evaluate_params(gpu_id, capacity, batch_size, lr, dropout, embedding_dim, model_name):
    cmd = [
        "python", "3_3_benchmark.py",
        "--capacity", str(capacity),
        "--model_name", model_name,
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--dropout", str(dropout),
        "--embedding_dim", str(embedding_dim),
        "--gpu_id", str(gpu_id)  # 在3_3_benchmark.py中用argparse接收
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception as e:
        print(f"[Error] {model_name} cap={capacity} batch_size={batch_size} 执行失败: {e}")
        return np.inf

    if result.returncode != 0 or "RuntimeError" in result.stderr or "CUDA out of memory" in result.stderr:
        err_msg = result.stderr if result.stderr else result.stdout
        cap_clean = str(capacity).replace(".0","")
        error_file = f"./outputs/{model_name}_{cap_clean}_error.txt"
        with open(error_file, "w") as f:
            f.write(f"超参数: batch_size={batch_size}, lr={lr}, dropout={dropout}, emb={embedding_dim}\n\n")
            f.write("===== 错误信息 =====\n")
            f.write(err_msg)
        print(f"[Error] {model_name} cap={capacity} batch_size={batch_size} 出错，日志已写入 {error_file}")
        return np.inf

    for line in result.stdout.splitlines()[::-1]:
        parts = line.strip().split()
        if len(parts) == 3:
            try:
                rmse = float(parts[0])
                return rmse
            except ValueError:
                continue
    return np.inf

# =========================
# 每个容量训练函数（指定 GPU）
# =========================
def train_capacity_on_gpu(gpu_id, capacity, best_params, model_name):
    cap_clean = str(capacity).replace(".0","")
    cmd = [
        "python", "3_3_benchmark.py",
        "--capacity", str(capacity),
        "--model_name", model_name,
        "--batch_size", str(best_params["batch_size"]),
        "--lr", str(best_params["lr"]),
        "--dropout", str(best_params["dropout"]),
        "--embedding_dim", str(best_params["embedding_dim"]),
        "--gpu_id", str(gpu_id)  # 在3_3_benchmark.py中用argparse接收
    ]
    output_file = f"./outputs/{model_name}_benchmark_{cap_clean}.txt"
    with open(output_file, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True, check=False)

    if result.returncode != 0 or "CUDA out of memory" in result.stderr:
        error_file = f"./outputs/{model_name}_{cap_clean}_error.txt"
        with open(error_file, "w") as f:
            f.write(f"===== 训练失败 =====\n")
            f.write(result.stderr)
        print(f"[OOM/Error] {model_name} cap={capacity} 训练失败，日志已写入 {error_file}")
    else:
        print(f"[GPU {gpu_id}] {model_name} cap={capacity} 训练完成，结果已保存到 {output_file}")

# =========================
# 主程序
# =========================
def main():
    num_gpus = torch.cuda.device_count()

    for model_name in model_names:
        print(f"\n========== 模型 {model_name} ==========")
        cache_file = f"./cache/best_params/{model_name}.txt"

        # 超参数搜索
        if os.path.exists(cache_file) and os.path.getsize(cache_file) > 0:
            with open(cache_file, "r") as f:
                best_params = json.load(f)
            print(f"加载已有超参数: {best_params}")
        else:
            print(f"未找到缓存文件，开始搜索 {model_name} 的最佳超参数...")
            best_params = default_params.copy()

            # 1. batch_size
            best_rmse = np.inf
            for bs in tqdm(batch_sizes, desc=f'{model_name} search batch_size'):
                rmse = evaluate_params(single_gpu_id, search_capacity, bs, best_params["lr"], best_params["dropout"], best_params["embedding_dim"], model_name)
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params["batch_size"] = bs

            # 2. learning_rate
            best_rmse = np.inf
            for lr in tqdm(learning_rates, desc=f'{model_name} search lr'):
                rmse = evaluate_params(single_gpu_id, search_capacity, best_params["batch_size"], lr, best_params["dropout"], best_params["embedding_dim"], model_name)
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params["lr"] = lr

            # 3. dropout
            best_rmse = np.inf
            for do in tqdm(dropouts, desc=f'{model_name} search dropout'):
                rmse = evaluate_params(single_gpu_id, search_capacity, best_params["batch_size"], best_params["lr"], do, best_params["embedding_dim"], model_name)
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params["dropout"] = do

            # 4. embedding_dim
            best_rmse = np.inf
            for emb in tqdm(embedding_dims, desc=f'{model_name} search embedding_dim'):
                rmse = evaluate_params(single_gpu_id, search_capacity, best_params["batch_size"], best_params["lr"], best_params["dropout"], emb, model_name)
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params["embedding_dim"] = emb

            print(f"{model_name} 最佳超参数:", best_params)

            # 保存超参数
            with open(cache_file, "w") as f:
                json.dump(best_params, f, indent=2)

        # === 跑训练 ===
        if use_gpus:
            # 多 GPU 并行
            assert num_gpus >= len(capacities), "至少需要与容量数量相等的 GPU"
            processes = []
            try:
                for gpu_id, cap in enumerate(capacities):
                    p = mp.Process(target=train_capacity_on_gpu, args=(gpu_id, cap, best_params, model_name))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()

            except KeyboardInterrupt:
                print("Ctrl+C detected, terminating all child processes...")
                for p in processes:
                    if p.is_alive():
                        p.terminate()
                for p in processes:
                    p.join()
                print("All child processes terminated.")
        else:
            # 单 GPU 串行
            for cap in capacities:
                train_capacity_on_gpu(single_gpu_id, cap, best_params, model_name)

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
