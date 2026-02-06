from transformers import AutoTokenizer
import os

class HttpTokenizer:
    def __init__(self, max_len=128):
        self.model_name = "JackTech/SecureBERT"
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def encode(self, text):
        """
        Encodes text into a dictionary of tensors: {'input_ids': ..., 'attention_mask': ...}
        """
        return self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
    
    # We keep these for compatibility but they are largely unused now
    def train(self, files):
        pass

    def save(self, path):
        # Saving just means saving the pretrained config usually, 
        # but AutoTokenizer is loaded by name mostly.
        # We can save to local dir.
        self.tokenizer.save_pretrained(os.path.dirname(path))

    def load(self, path):
        # We can load from local dir if saved there
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(path))
        except:
            # Fallback to web
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
