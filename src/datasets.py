import os
import numpy as np
import torch
from typing import Tuple
from glob import glob

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", num_subjects: int = 4) -> None:
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        
        self.split = split
        self.data_dir = data_dir
        self.num_classes = 1854
        self.num_subjects = num_subjects  # 被験者の数
        self.num_samples = len(glob(os.path.join(data_dir, f"{split}_X", "*.npy")))

        # 訓練データセットで計算・保存されている統計量を読み込む
        self.mean = np.load(os.path.join(data_dir, "train_mean.npy")).astype(np.float32)
        self.std = np.load(os.path.join(data_dir, "train_std.npy")).astype(np.float32)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, i):
        X_path = os.path.join(self.data_dir, f"{self.split}_X", str(i).zfill(5) + ".npy")
        X = np.load(X_path).astype(np.float32)
        
        subject_idx_path = os.path.join(self.data_dir, f"{self.split}_subject_idxs", str(i).zfill(5) + ".npy")
        subject_idx = np.load(subject_idx_path).astype(np.int64)  # 整数型に変換
        
        if self.split in ["train", "val"]:
            y_path = os.path.join(self.data_dir, f"{self.split}_y", str(i).zfill(5) + ".npy")
            y = np.load(y_path).astype(np.int64)  # ラベルを整数型に変換
            
            # データの標準化
            X = self.standardize_data(X)

            return torch.from_numpy(X), torch.from_numpy(y), torch.tensor(subject_idx)
        else:
            # テストデータも標準化する
            X = self.standardize_data(X)
            return torch.from_numpy(X), torch.tensor(subject_idx)
        
    @property
    def num_channels(self) -> int:
        return np.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")).shape[0]
    
    @property
    def seq_len(self) -> int:
        return np.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")).shape[1]

    def standardize_data(self, data):
        """計算された平均と標準偏差を使用してデータを標準化"""
        return (data - self.mean[:, None]) / self.std[:, None]
