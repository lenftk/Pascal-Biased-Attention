import os
import pickle
import torch
import numpy as np
import argparse
from tqdm import tqdm
from tokenizers import Tokenizer
from datasets import load_dataset
from codecarbon import track_emissions
from config.config import ModelConfig, TrainConfig
from src.model import PascalLanguageModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_ppl_test_data(dataset_name, tokenizer, block_size):
    if dataset_name == 'wikitext103':
        dataset = load_dataset('wikitext', 'wikitext-103-v1', split='validation')
        text = "\n".join(ex['text'] for ex in dataset if ex['text'].strip())
    else:
        raise ValueError("Unsupported dataset")
    encoded_ids = tokenizer.encode(text).ids
    return np.array(encoded_ids, dtype=np.uint32)

@track_emissions
def evaluate_ppl(model, tokenizer, dataset_name):
    test_data = get_ppl_test_data(dataset_name, tokenizer, model.config.block_size)
    seq_len = model.config.block_size
    nlls = []
    with torch.no_grad():
        for i in tqdm(range(0, len(test_data) - seq_len, seq_len)):
            inputs = torch.from_numpy(test_data[i:i+seq_len].astype(np.int64)).unsqueeze(0).to(device)
            targets = torch.from_numpy(test_data[i+1:i+1+seq_len].astype(np.int64)).unsqueeze(0).to(device)
            _, loss = model(inputs, targets)
            nlls.append(loss)
    print(f"PPL: {torch.exp(torch.stack(nlls).mean()).item():.4f}")

def main(args):
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    with open(args.meta_path, 'rb') as f: meta = pickle.load(f)
    
    m_config = ModelConfig()
    m_config.vocab_size = meta['vocab_size']
    model = PascalLanguageModel(m_config, TrainConfig())
    checkpoint = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    if args.dataset == 'wikitext103':
        evaluate_ppl(model, tokenizer, args.dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='wikitext103')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoint/model_checkpoint.pt')
    parser.add_argument('--tokenizer_path', type=str, default='./data/bpe_tokenizer.json')
    parser.add_argument('--meta_path', type=str, default='./data/meta.pkl')
    args = parser.parse_args()
    main(args)