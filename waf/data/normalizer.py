import re

class RequestNormalizer:
    def __init__(self):
        # Patterns to replace with placeholders
        self.uuid_pattern = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', re.IGNORECASE)
        self.email_pattern = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
        self.number_pattern = re.compile(r'\d+')
        
        # Specific cleaning for Paths
        # Replace numbers in paths with {ID}
        # e.g. /api/users/123 -> /api/users/{ID}
    
    def normalize(self, log_entry: dict) -> str:
        """
        Converts a parsed log entry into a canonical string representation for the model.
        Format: [METHOD] [URI] [BODY]
        """
        method = log_entry.get('method', 'GET')
        uri = log_entry.get('uri', '/')
        body = log_entry.get('body', '')

        # Normalize URI
        # 1. Lowercase? Maybe not, case sensitive payloads exist.
        # 2. Replace UUIDs
        uri = self.uuid_pattern.sub('{UUID}', uri)
        # 3. Replace numeric IDs in path segments
        # We want to keep "v1" but mask "12345"
        # Simple heuristic: if a segment is pure digits, replace with {ID}
        segments = uri.split('/')
        norm_segments = []
        for seg in segments:
            if seg.isdigit():
                norm_segments.append('{ID}')
            else:
                norm_segments.append(seg)
        uri = '/'.join(norm_segments)

        # Normalize Body
        if body:
            # Simple normalization for body
            body = self.uuid_pattern.sub('{UUID}', body)
            body = self.email_pattern.sub('{EMAIL}', body)
            # body = self.number_pattern.sub('{NUM}', body) # Be careful with this, might lose context
        
        # Concatenate parts
        # Using special tokens to separate fields could be useful for the Tokenizer later
        # [METHOD] URI [BODY] BodyContent
        full_request = f"{method} {uri}"
        if body:
            full_request += f" {body}"
        
        return full_request
