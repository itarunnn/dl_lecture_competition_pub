import numpy as np
import os

# ファイルパス
train_mean_path = 'data/train_mean.npy'
train_std_path = 'data/train_std.npy'

# ファイルの存在を確認し、内容を読み込む
if os.path.exists(train_mean_path) and os.path.exists(train_std_path):
    train_mean = np.load(train_mean_path)
    train_std = np.load(train_std_path)
    print("Train Mean Shape:", train_mean.shape)
    print("Train Std Shape:", train_std.shape)
else:
    print("必要なファイルが存在しません。")