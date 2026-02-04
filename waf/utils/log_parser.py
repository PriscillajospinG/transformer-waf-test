import re
import json
from urllib.parse import unquote

class NginxLogParser:
    def __init__(self):
        # Log format: '$remote_addr - $remote_user [$time_local] "$request" $status $body_bytes_sent "$http_referer" "$http_user_agent" "$request_body"'
        # Example: 172.18.0.1 - - [04/Feb/2026:10:00:00 +0000] "GET / HTTP/1.1" 200 123 "-" "Mozilla/5.0" "-"
        
        # Regex explanation:
        # (?P<ip>[\d\.]+)           -> IP address
        # \s-\s                     -> " - " separator
        # (?P<user>\S+)             -> Remote user (often "-")
        # \s\[(?P<time>[^\]]+)\]    -> [Time]
        # \s"(?P<request>[^"]+)"    -> "Method URI Protocol"
        # \s(?P<status>\d+)         -> Status Code
        # \s(?P<bytes>\d+)          -> Bytes Sent
        # \s"(?P<referer>[^"]*)"    -> Referer
        # \s"(?P<ua>[^"]*)"         -> User Agent
        # \s"(?P<body>.*)"          -> Request Body (Last part, can be empty or "-")
        
        self.log_pattern = re.compile(
            r'(?P<ip>[\d\.]+)\s-\s(?P<user>\S+)\s\[(?P<time>[^\]]+)\]\s"(?P<request>[^"]+)"\s(?P<status>\d+)\s(?P<bytes>\d+)\s"(?P<referer>[^"]*)"\s"(?P<ua>[^"]*)"\s"(?P<body>.*)"'
        )

    def parse_line(self, line: str) -> dict:
        """Parses a single line of Nginx log into a structured dictionary."""
        match = self.log_pattern.match(line)
        if not match:
            return None
        
        data = match.groupdict()
        
        # Deconstruct request line "GET /foo?bar=1 HTTP/1.1"
        request_parts = data['request'].split(' ')
        if len(request_parts) >= 2:
            method = request_parts[0]
            uri = request_parts[1]
        else:
            method = "UNKNOWN"
            uri = data['request']

        # Normalize parsing
        parsed = {
            "ip": data['ip'],
            "timestamp": data['time'],
            "method": method,
            "uri": unquote(uri),  # Decode URI encoded characters
            "status": int(data['status']),
            "user_agent": data['ua'],
            "body": data['body'] if data['body'] != "-" else "",
            "referer": data['referer']
        }
        
        return parsed

    def parse_file(self, file_path: str):
        """Generator that yields parsed log entries from a file."""
        with open(file_path, 'r') as f:
            for line in f:
                parsed = self.parse_line(line.strip())
                if parsed:
                    yield parsed
