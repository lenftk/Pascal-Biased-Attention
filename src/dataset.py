import os
import re
import pickle
import random
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from config.config import DataConfig, ModelConfig

def prepare_data_main(config: DataConfig):
    def clean_text(text):
        text = re.sub(r'\n{2,}', '\n', text).strip()
        return text
    print(f"Starting download and processing for '{config.dataset_name} ({config.dataset_subset})'...")
    output_dir = os.path.dirname(config.clean_text_path)
    os.makedirs(output_dir, exist_ok=True)
    try:
        dataset = load_dataset(config.dataset_name, config.dataset_subset, split='train', streaming=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    with open(config.clean_text_path, 'w', encoding='utf-8') as outfile:
        count = 0
        for example in tqdm(dataset, desc="Processing and saving data", unit=" docs"):
            text = example.get('text', '')
            if text:
                cleaned_text = clean_text(text)
                if len(cleaned_text) > 50:
                    outfile.write(cleaned_text + "\n")
                    count += 1
    print(f"\nDataset processing complete! Saved {count} documents to '{config.clean_text_path}'.")

def run_memory_safe_tokenization(d_config: DataConfig, m_config: ModelConfig):
    print("--- Starting BPE tokenization and data split ---")
    if not os.path.exists(d_config.clean_text_path):
        print(f"Error: file '{d_config.clean_text_path}' not found.")
        return
    
    # Load or train tokenizer
    if os.path.exists(d_config.tokenizer_path):
        tokenizer = Tokenizer.from_file(d_config.tokenizer_path)
        print("Existing tokenizer loaded.")
    else:
        print("No existing tokenizer found. Training a new one...")
        with open(d_config.clean_text_path, 'r', encoding='utf-8') as f:
            text_iterator = (line for line in f)
            tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            tokenizer.normalizer = normalizers.NFKC()
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            trainer = BpeTrainer(vocab_size=m_config.vocab_size, min_frequency=2, special_tokens=["[PAD]", "[UNK]", "[EOS]"])
            tokenizer.train_from_iterator(text_iterator, trainer=trainer)
            tokenizer.save(d_config.tokenizer_path)
            print(f"Tokenizer training complete and saved: {d_config.tokenizer_path}")
    
    # Encode and save to binary
    data_dir = os.path.dirname(d_config.clean_text_path)
    train_bin_path = os.path.join(data_dir, 'train.bin')
    val_bin_path = os.path.join(data_dir, 'val.bin')
    train_token_count, val_token_count = 0, 0
    
    with open(d_config.clean_text_path, 'r', encoding='utf-8') as f_in, \
         open(train_bin_path, 'wb') as f_train, \
         open(val_bin_path, 'wb') as f_val:
        for line in tqdm(f_in, desc="Encoding and splitting file", unit=" lines"):
            if not line.strip(): continue
            encoded = tokenizer.encode(line)
            ids = encoded.ids
            if random.random() < d_config.test_size:
                f_val.write(np.array(ids, dtype=np.uint32).tobytes())
                val_token_count += len(ids)
            else:
                f_train.write(np.array(ids, dtype=np.uint32).tobytes())
                train_token_count += len(ids)
                
    meta = {'vocab_size': tokenizer.get_vocab_size(), 'tokenizer_path': d_config.tokenizer_path, 'eos_token_id': tokenizer.token_to_id("[EOS]")}
    with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
        
    print(f"\nTokenization and split complete.\nTrain data: {train_token_count:,} tokens\nVal data: {val_token_count:,} tokens")