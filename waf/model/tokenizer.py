from tokenizers import Tokenizer
from tokenizers.models import BPE, WordLevel
from tokenizers.trainers import BpeTrainer, WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
import os

class HttpTokenizer:
    def __init__(self, vocab_size=30000, model_type="bpe"):
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.tokenizer = None
        
        if model_type == "bpe":
            self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            self.trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"], vocab_size=vocab_size)
        else:
            self.tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
            self.trainer = WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"], vocab_size=vocab_size)
            
        self.tokenizer.pre_tokenizer = Whitespace()

    def train(self, files):
        """Trains the tokenizer on a list of files containing normalized requests."""
        self.tokenizer.train(files, self.trainer)

    def save(self, path):
        self.tokenizer.save(path)

    def load(self, path):
        self.tokenizer = Tokenizer.from_file(path)

    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, ids):
        return self.tokenizer.decode(ids)
