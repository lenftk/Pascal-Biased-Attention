import os
import pickle
import torch
import torch.nn.functional as F
import argparse
from tokenizers import Tokenizer
from config.config import ModelConfig, TrainConfig
from src.model import PascalLanguageModel

def main(args):
    if not os.path.exists(args.ckpt_path):
        print(f"Checkpoint file not found: {args.ckpt_path}")
        return

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    with open(args.meta_path, 'rb') as f: meta = pickle.load(f)

    m_config = ModelConfig()
    m_config.vocab_size = meta['vocab_size']
    
    # Load model (set map_location='cpu' for CPU inference)
    model = PascalLanguageModel(m_config, TrainConfig())
    checkpoint = torch.load(args.ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    print(f"Prompt: {args.prompt}")
    input_ids = tokenizer.encode(args.prompt).ids
    idx = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)

    generated_ids = []
    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            idx_cond = idx if idx.size(1) <= m_config.block_size else idx[:, -m_config.block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / args.temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            generated_ids.append(idx_next.item())
            
    print("Generated:", tokenizer.decode(generated_ids))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default='Hello, my name is')
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--ckpt_path', type=str, default='./checkpoint/model_checkpoint_cpu.pt')
    parser.add_argument('--tokenizer_path', type=str, default='./data/bpe_tokenizer.json')
    parser.add_argument('--meta_path', type=str, default='./data/meta.pkl')
    args = parser.parse_args()
    main(args)