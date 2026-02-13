import os
import time
import math
import pickle
import numpy as np
import torch
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from config.config import DataConfig, ModelConfig, TrainConfig
from src.dataset import prepare_data_main, run_memory_safe_tokenization
from src.model import PascalLanguageModel

def train_model():
    m_config = ModelConfig()
    t_config = TrainConfig()
    os.makedirs(t_config.ckpt_dir, exist_ok=True)
    
    device = t_config.device
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load data
    data_dir = t_config.data_dir
    train_data = np.fromfile(os.path.join(data_dir, 'train.bin'), dtype=np.uint32)
    val_data = np.fromfile(os.path.join(data_dir, 'val.bin'), dtype=np.uint32)
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    m_config.vocab_size = meta['vocab_size']
    print(f"Data loaded. Vocab size: {m_config.vocab_size}")

    model = PascalLanguageModel(m_config, t_config).to(device)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    try:
        model = torch.compile(model)
        print("torch.compile applied.")
    except Exception as e:
        print(f"torch.compile failed: {e}. Running in standard mode.")
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=t_config.learning_rate, weight_decay=t_config.weight_decay, betas=(t_config.beta1, t_config.beta2))
    scaler = GradScaler(enabled=(device_type == 'cuda'))

    def get_lr(it):
        if it < t_config.warmup_iters: return t_config.learning_rate * it / t_config.warmup_iters
        if it > t_config.lr_decay_iters: return t_config.learning_rate * t_config.min_lr_ratio
        decay_ratio = (it - t_config.warmup_iters) / (t_config.lr_decay_iters - t_config.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        min_lr = t_config.learning_rate * t_config.min_lr_ratio
        return min_lr + coeff * (t_config.learning_rate - min_lr)

    iter_num, best_val_loss = 0, 1e9

    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - m_config.block_size, (t_config.batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i+m_config.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i+1:i+1+m_config.block_size].astype(np.int64)) for i in ix])
        if device_type == 'cuda':
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(t_config.eval_iters)
            pbar = tqdm(range(t_config.eval_iters), desc=f"Evaluating {split}", leave=False, unit="batch")
            for k in pbar:
                X, Y = get_batch(split)
                with autocast(device_type=device_type, dtype=torch.bfloat16):
                    _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    X, Y = get_batch('train')
    start_time = time.time()
    print(f"Starting training... (Total {t_config.max_iters} steps)")
    main_pbar = tqdm(range(iter_num, t_config.max_iters), desc="Training", unit="iter")

    for iter_num in main_pbar:
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups: param_group['lr'] = lr

        if iter_num > 0 and iter_num % t_config.eval_interval == 0:
            losses = estimate_loss()
            elapsed_time = time.time() - start_time
            start_time = time.time()
            
            uncompiled_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            bias_info = uncompiled_model.get_bias_info()
            
            main_pbar.set_postfix(train_loss=f"{losses['train']:.3f}", val_loss=f"{losses['val']:.3f}")
            print(f"\n[Iter {iter_num:5d}] Train: {losses['train']:.4f} | Val: {losses['val']:.4f} | LR: {lr:.2e} | Time: {elapsed_time:.1f}s")
            print(f"  Pascal Bias: {bias_info['bias_strength']:.4f} | Penalty: {bias_info['skip_penalty']:.4f}")
            
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                checkpoint = {
                    'model': uncompiled_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss
                }
                torch.save(checkpoint, t_config.ckpt_path)
                print(f"  ==> Checkpoint saved: {t_config.ckpt_path}")

        optimizer.zero_grad(set_to_none=True)
        for micro_step in range(t_config.gradient_accumulation_steps):
            with autocast(device_type=device_type, dtype=torch.bfloat16):
                _, loss = model(X, Y)
                loss = loss / t_config.gradient_accumulation_steps
            X, Y = get_batch('train')
            scaler.scale(loss).backward()
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), t_config.grad_clip)
        scaler.step(optimizer)
        scaler.update()

    print("Training complete!")

if __name__ == '__main__':
    d_config = DataConfig()
    m_config = ModelConfig()
    
    # 1. Prepare data
    if not os.path.exists(d_config.clean_text_path):
        prepare_data_main(d_config)
    
    # 2. Tokenization
    data_dir = d_config.data_dir
    required_files = [os.path.join(data_dir, 'train.bin'), os.path.join(data_dir, 'val.bin'), os.path.join(data_dir, 'meta.pkl')]
    if not all(os.path.exists(p) for p in required_files):
        run_memory_safe_tokenization(d_config, m_config)
        
    # 3. Start training
    train_model()