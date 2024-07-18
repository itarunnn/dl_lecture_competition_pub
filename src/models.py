import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import torch.nn.init as init

def initialize_weights(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class BasicConvClassifier(nn.Module):
    def __init__(self, num_classes: int, seq_len: int, in_channels: int, num_subjects: int, hid_dim: int = 128, p_drop: float = 0.1, emb_dim: int = 64) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim, p_drop=p_drop),
            ConvBlock(hid_dim, hid_dim, p_drop=p_drop),
            ConvBlock(hid_dim, hid_dim, p_drop=p_drop),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim + emb_dim, 512),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(512, num_classes)
        )

        # 被験者情報をエンベディングに変換する層
        self.subject_embedding = nn.Embedding(num_subjects, emb_dim)

        # 重み初期化を適用
        self.apply(initialize_weights)

    def forward(self, X: torch.Tensor, subject_idx: torch.Tensor) -> torch.Tensor:
        X = self.blocks(X)
        X = self.head[0:2](X)  # Apply only AdaptiveAvgPool1d and Rearrange
        subject_emb = F.gelu(self.subject_embedding(subject_idx))
        combined = torch.cat((X, subject_emb), dim=1)
        return self.head[2:](combined)  # Apply remaining layers of head

class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size: int = 3, p_drop: float = 0.1) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        return self.dropout(X)
