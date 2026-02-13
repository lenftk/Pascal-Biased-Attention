import torch
import os

ckpt_path = './checkpoint/model_checkpoint.pt'

def check_checkpoint(path):
    if not os.path.exists(path):
        print(f"Checkpoint not found: {path}")
        return
    
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        print(f"--- Checkpoint Info ---")
        print(f"Iter Num: {checkpoint.get('iter_num', 'N/A')}")
        print(f"Best Val Loss: {checkpoint.get('best_val_loss', 'N/A')}")
        print("Model keys loaded successfully.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

if __name__ == "__main__":
    check_checkpoint(ckpt_path)