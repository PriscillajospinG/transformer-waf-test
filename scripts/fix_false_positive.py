import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys

# Add parent dir
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from waf.model.transformer import WAFTransformer
from waf.model.tokenizer import HttpTokenizer
from waf.train.train_pipeline import WAFDataset

def fix_false_positive():
    print("--- Fixing False Positive: 'apple' search ---")
    
    TOKENIZER_PATH = "waf/model/weights/tokenizer.json"
    MODEL_PATH = "waf/model/weights/waf_model.pth"
    device = torch.device("cpu")
    
    # 1. Load
    tokenizer = HttpTokenizer()
    tokenizer.load(TOKENIZER_PATH)
    
    model = WAFTransformer(vocab_size=1000, d_model=64, num_layers=2, num_classes=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.train()
    
    # 2. Data
    # The false positive request
    # NOTE: We must match the preprocessing in main.py exactly!
    # main.py: text = f"{method} {uri_norm}" where uri_norm replaces digits with {ID}
    # "GET /rest/products/search?q=apple" -> No digits to replace
    text = "GET /rest/products/search?q=apple"
    
    # Benign data (The fix + Replay)
    benign_samples = [
        "GET /rest/products/search?q=apple",
        "GET /",
        "GET /api/Products"
    ] * 10 
    
    # Malicious data (To maintain boundary)
    malicious_samples = [
        "GET /rest/products/search?q=' OR 1=1 --",
        "GET /rest/products/search?q=<script>alert(1)</script>",
        "GET /etc/passwd",
        "GET /api/Users?email=' OR '1'='1"
    ] * 5 # 4 * 5 = 20 samples
    
    data = benign_samples + malicious_samples
    labels = [0] * len(benign_samples) + [1] * len(malicious_samples)
    
    dataset = WAFDataset(data, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # 3. Train
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Higher LR for immediate fix
    
    print("Fine-tuning...")
    for epoch in range(5):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
    # 4. Verify in-memory
    model.eval()
    with torch.no_grad():
        encoded = tokenizer.encode(text)
        ids = encoded.ids + [0]*(64 - len(encoded.ids))
        inp = torch.tensor([ids], dtype=torch.long)
        out = model(inp)
        prob = torch.softmax(out, dim=1)
        print(f"New Prediction for '{text}': {prob[0].tolist()} (Class: {torch.argmax(prob).item()})")
        
    # 5. Save
    torch.save(model.state_dict(), MODEL_PATH)
    print("Model corrected and saved.")
    
    # 6. Restart Container
    print("Restarting WAF service to load new weights...")
    os.system("docker restart waf-service")

if __name__ == "__main__":
    fix_false_positive()
