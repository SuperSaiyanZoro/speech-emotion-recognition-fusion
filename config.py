import torch
from dataclasses import dataclass

@dataclass
class Config:
    seed: int = 42
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-3
    num_classes: int = 6
    wav2vec_dim: int = 768
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

config = Config()
