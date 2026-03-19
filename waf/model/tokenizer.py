import logging
import os

from transformers import AutoTokenizer

logger = logging.getLogger("waf-service")


class HttpTokenizer:
    def __init__(self, max_len=256):
        if max_len < 64 or max_len > 512:
            raise ValueError("max_len must be between 64 and 512")

        self.model_name = "bert-base-uncased"
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def _normalize_text(self, text):
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        # Drop null bytes that can break downstream processing.
        return text.replace("\x00", "")

    def encode(self, text):
        text = self._normalize_text(text)
        try:
            encoded = self.tokenizer(
                text,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            # Lightweight truncation signal for observability.
            estimated_tokens = len(text.split())
            if estimated_tokens > self.max_len:
                logger.warning(
                    "Tokenizer truncation applied: estimated_tokens=%s max_len=%s",
                    estimated_tokens,
                    self.max_len,
                )
            return encoded
        except Exception as exc:
            logger.error("Tokenizer encode failed: %s", exc)
            # Return safe fallback encoding of empty request.
            return self.tokenizer(
                "",
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

    def train(self, files):
        return None

    def save(self, path):
        self.tokenizer.save_pretrained(os.path.dirname(path))

    def load(self, path):
        local_dir = os.path.dirname(path)
        if os.path.exists(local_dir):
            self.tokenizer = AutoTokenizer.from_pretrained(local_dir)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
