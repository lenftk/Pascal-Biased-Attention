import os
from dataclasses import dataclass
import torch

@dataclass
class DataConfig:
    dataset_name: str = "wikitext"
    dataset_subset: str = "wikitext-103-raw-v1"
    data_dir: str = "data"
    clean_text_path: str = os.path.join(data_dir, "input_cleaned.txt")
    tokenizer_path: str = os.path.join(data_dir, "bpe_tokenizer.json")
    test_size: float = 0.05
    random_state: int = 42

@dataclass
class ModelConfig:
    block_size: int = 256
    vocab_size: int = 8192
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    pascal_n: int = 3
    bias: bool = False
    dropout: float = 0.1

@dataclass
class TrainConfig:
    data_dir: str = "data"
    ckpt_dir: str = "checkpoint"
    ckpt_path: str = os.path.join(ckpt_dir, "model_checkpoint.pt")
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size: int = 64
    gradient_accumulation_steps: int = 2
    max_iters: int = 20000
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_iters: int = 2000
    bias_warmup_iters: int = 4000
    lr_decay_iters: int = 8000
    min_lr_ratio: float = 0.1
    eval_interval: int = 250
    eval_iters: int = 100
    log_interval: int = 10