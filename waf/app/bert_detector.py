"""
BERT Detection Engine
SecureBERT-based malicious request detection
"""

import torch
import logging
from typing import Tuple, Dict, Optional, List
from app.config import WAFConfig

logger = logging.getLogger("bert-detector")

class BERTDetector:
    """BERT-based detector for malicious requests"""
    
    def __init__(self, model=None, tokenizer=None, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cpu")
        self.model_loaded = model is not None and tokenizer is not None
    
    def set_model(self, model, tokenizer):
        """Set the model and tokenizer"""
        self.model = model
        self.tokenizer = tokenizer
        self.model_loaded = True
    
    def detect(self, text: str) -> Tuple[float, float, float]:
        """
        Detect if text is malicious using BERT
        Returns: (benign_prob, malicious_prob, confidence)
        """
        if not self.model_loaded or self.model is None or self.tokenizer is None:
            logger.warning("Model not loaded, returning neutral scores")
            return 0.5, 0.5, 0.5
        
        try:
            # Tokenize
            encoding = self.tokenizer.encode(text)
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs, dim=1)
                
                benign_prob = float(probs[0][0].detach().cpu().item())
                malicious_prob = float(probs[0][1].detach().cpu().item())
                confidence = max(benign_prob, malicious_prob)
            
            return benign_prob, malicious_prob, confidence
        
        except Exception as e:
            logger.error(f"BERT detection error: {e}")
            return 0.5, 0.5, 0.5
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get embedding vector from BERT hidden states
        """
        if not self.model_loaded or self.model is None or self.tokenizer is None:
            return None
        
        try:
            encoding = self.tokenizer.encode(text)
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            with torch.no_grad():
                # Get hidden states before classification head
                outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                
                # Use [CLS] token from last hidden state
                hidden_state = outputs.hidden_states[-1]  # Last layer
                cls_embedding = hidden_state[0, 0, :].detach().cpu().numpy()
                
                return cls_embedding.tolist()
        
        except Exception as e:
            logger.error(f"Embedding extraction error: {e}")
            return None
    
    def batch_detect(self, texts: List[str]) -> List[Tuple[float, float, float]]:
        """
        Detect multiple texts in batch (more efficient)
        Returns: list of (benign_prob, malicious_prob, confidence)
        """
        results = []
        for text in texts:
            results.append(self.detect(text))
        return results

# Global BERT detector instance
bert_detector = BERTDetector()
