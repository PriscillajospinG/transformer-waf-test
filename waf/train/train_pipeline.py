import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import os
import sys

# Add parent dir
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from waf.model.transformer import WAFTransformer
from waf.model.tokenizer import HttpTokenizer

# --- Synthetic Data Generation ---
BENIGN_TEMPLATES = [
    "GET / HTTP/1.1",
    "GET /favicon.ico HTTP/1.1",
    "GET /api/Users HTTP/1.1",
    "GET /api/Products/{ID} HTTP/1.1",
    "POST /api/Login HTTP/1.1",
    "GET /assets/js/main.js HTTP/1.1",
    "GET /rest/products/search?q=apple HTTP/1.1",
    "GET /rest/products/search?q=banana HTTP/1.1",
    "GET /#/contact HTTP/1.1",
]

MALICIOUS_TEMPLATES = [
    "GET /rest/products/search?q=' OR 1=1 -- HTTP/1.1",
    "GET /rest/products/search?q=<script>alert(1)</script> HTTP/1.1",
    "GET /etc/passwd HTTP/1.1",
    "POST /api/Login ' OR '1'='1 HTTP/1.1",
    "GET /../../../etc/shadow HTTP/1.1",
    "GET /?cmd=cat /etc/passwd HTTP/1.1",
    "GET /dvwa/vulnerabilities/sqli/?id=%27+or+1%3D1+--+&Submit=Submit HTTP/1.1",
]

def generate_synthetic_data(num_samples=2000):
    data = []
    labels = []
    
    # Benign (Label 0)
    for _ in range(num_samples // 2):
        base = random.choice(BENIGN_TEMPLATES)
        # Add some random variations
        if "{ID}" in base:
            base = base.replace("{ID}", str(random.randint(1, 1000)))
        data.append(base)
        labels.append(0)
        
    # Malicious (Label 1)
    for _ in range(num_samples // 2):
        base = random.choice(MALICIOUS_TEMPLATES)
        data.append(base)
        labels.append(1)
        
    return data, labels

# --- Dataset ---
class WAFDataset(Dataset):
    def __init__(self, requests, labels, tokenizer, max_len=64):
        self.requests = requests
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.requests)

    def __getitem__(self, idx):
        text = self.requests[idx]
        label = self.labels[idx]
        
        encoded = self.tokenizer.encode(text)
        ids = encoded.ids
        
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        else:
            ids = ids + [0] * (self.max_len - len(ids))
            
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# --- Training Loop ---
def train_pipeline():
    print("1. Generating Synthetic Data...")
    texts, labels = generate_synthetic_data(4000)
    
    # Dump to text file for tokenizer training
    with open("dataset_synthetic.txt", "w") as f:
        for t in texts:
            f.write(t + "\n")
            
    print("2. Training Tokenizer...")
    tokenizer = HttpTokenizer(vocab_size=1000)
    tokenizer.train(["dataset_synthetic.txt"])
    
    os.makedirs("waf/model/weights", exist_ok=True)
    tokenizer.save("waf/model/weights/tokenizer.json")
    print("Tokenizer saved.")
    
    print("3. Training Model...")
    dataset = WAFDataset(texts, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = WAFTransformer(vocab_size=1000, d_model=64, num_layers=2, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 5
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss/len(dataloader):.4f} | Acc: {100.*correct/total:.2f}%")
        
    print("4. Saving Model...")
    torch.save(model.state_dict(), "waf/model/weights/waf_model.pth")
    print("Model saved to waf/model/weights/waf_model.pth")

if __name__ == "__main__":
    train_pipeline()
