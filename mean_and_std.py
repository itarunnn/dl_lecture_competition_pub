import os
import numpy as np
from glob import glob

def calculate_and_save_statistics(data_dir, split="train"):
    data_paths = glob(os.path.join(data_dir, f"{split}_X", "*.npy"))

    # 初期化
    mean_sum = None
    var_sum = None
    count = 0

    # バッチごとに処理
    for path in data_paths:
        data = np.load(path)
        if mean_sum is None:
            mean_sum = np.zeros(data.shape[0])  # (チャンネル数,)
            var_sum = np.zeros(data.shape[0])  # (チャンネル数,)

        mean_sum += data.sum(axis=1)
        var_sum += (data ** 2).sum(axis=1)
        count += data.shape[1]

    # 平均と分散の計算
    mean = mean_sum / count
    variance = (var_sum / count) - (mean ** 2)
    std = np.sqrt(variance)

    # 統計量を保存
    np.save(os.path.join(data_dir, f"{split}_mean.npy"), mean)
    np.save(os.path.join(data_dir, f"{split}_std.npy"), std)

calculate_and_save_statistics("data")
