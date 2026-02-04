import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
import os

# Add parent dir
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from waf.model.transformer import WAFTransformer
from waf.model.tokenizer import HttpTokenizer

class WAFDataset(Dataset):
    def __init__(self, requests, labels, tokenizer, max_len=128):
        self.requests = requests
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.requests)

    def __getitem__(self, idx):
        text = self.requests[idx]
        label = self.labels[idx]
        
        # Tokenize
        encoded = self.tokenizer.encode(text)
        ids = encoded.ids
        
        # Pad/Truncate
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        else:
            ids = ids + [0] * (self.max_len - len(ids)) # Assuming 0 is pad
            
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def train():
    TOKENIZER_PATH = "waf/model/tokenizer.json"
    MODEL_PATH = "waf/model/weights/waf_model.pth"
    
    # Load Tokenizer
    tokenizer = HttpTokenizer()
    if os.path.exists(TOKENIZER_PATH):
        tokenizer.load(TOKENIZER_PATH)
    else:
        print("Tokenizer not found. Run dataset generation first.")
        return

    # Load Data
    # For this demo, we'll try to load from the 'dataset.txt' but we need labels.
    # We will "Game" it a bit: we will recreate the dataset in memory during training for simplicity 
    # OR we rely on the files generated in `scripts/`.
    # Let's check if we can just read `dataset.txt` and assume it's mixed? No, we need labels.
    
    # Let's perform a quick re-generation in memory for training proof-of-concept
    # In a real system, we'd have properly labeled files.
    # I'll just hardcode some dummy data logic here if files are missing, 
    # but ideally we rely on the `dataset.txt` from `build_dataset.py`.
    
    # ACTUALLY, I will rely on reading 'benign.txt' and 'malicious.txt' which I will create.
    # I'll modify the `build_dataset.py` or just split logic later.
    # For now, let's assume `dataset.txt` has everything and we don't know labels -> Unsupervised?
    # User asked for "Supervised classification (benign vs malicious)".
    
    # Let's make this script robust: It will try to find 'benign.txt' and 'malicious.txt'
    # If not found, it will complain.
    
    benign_reqs = []
    malicious_reqs = []
    
    # Hack: use the generators directly to get strings if files don't exist?
    # No, that's messy.
    
    # Let's just create some dummy data here if we can't find files, for the sake of the script working.
    pass 

if __name__ == "__main__":
    # Updated plan: I'll write a better training script in the next step
    # that actually generates data on the fly or reads from specific files.
    print("Use the 'train_pipeline.py' instead.")
