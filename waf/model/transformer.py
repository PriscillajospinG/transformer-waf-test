import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class WAFTransformer(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Use a reputable SecureBERT model from Hugging Face
        # JackTech/SecureBERT is a common choice for this domain
        self.model_name = "JackTech/SecureBERT" 
        
        print(f"Loading {self.model_name}...")
        self.bert = AutoModel.from_pretrained(self.model_name)
        
        # Freezing layers (Optional): Unfreeze for fine-tuning entire model
        # For better performance on small data, we often fine-tune all layers.
        
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None):
        # BERT forward pass
        # output[0] = last_hidden_state, output[1] = pooler_output (if available)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # We use the [CLS] token embedding (first token) for classification
        last_hidden_state = outputs.last_hidden_state
        cls_token = last_hidden_state[:, 0, :]
        
        x = self.drop(cls_token)
        logits = self.fc(x)
        return logits
