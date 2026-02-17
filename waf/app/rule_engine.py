"""
Rule-based Attack Detection Engine
Fast, interpretable detection based on known patterns
"""

import re
from typing import Tuple, List, Dict

class RuleEngine:
    """Rule-based detector for common attack patterns"""
    
    # SQL Injection patterns
    SQL_INJECTION_PATTERNS = [
        r"(\bUNION\b.*\bSELECT\b|\bSELECT\b.*\bFROM\b)",
        r"(DROP\s+TABLE|DELETE\s+FROM|INSERT\s+INTO|UPDATE\s+\w+\s+SET)",
        r"('[\s]*OR[\s]*'|'[\s]*OR[\s]*1[\s]*=|'\s*OR\s*'1'='1)",
        r"(;\s*DROP|;\s*DELETE|;\s*EXEC|;\s*EXECUTE)",
        r"(UNION\s+ALL\s+SELECT|UNION\s*/\*)",
        r"(\bXP_\w+|SP_\w+|SP_EXECUTESQL)",  # SQL Server procedures
        r"(SLEEP\s*\(|BENCHMARK\s*\()",  # Time-based SQLi  
        r"(',\s*\d+,\s*'|',\d+,')",  # Common UNION SELECT patterns
    ]
    
    # Cross-Site Scripting (XSS) patterns
    XSS_PATTERNS = [
        r"<\s*script[^>]*>",
        r"javascript\s*:",
        r"on(load|error|click|mouse\w+)\s*=",
        r"<\s*iframe[^>]*>",
        r"<\s*object[^>]*>",
        r"<\s*embed[^>]*>",
        r"eval\s*\(",
        r"expression\s*\(",
        r"vbscript\s*:",
        r"<\s*img[^>]*\s+src\s*=",
    ]
    
    # Command Injection patterns
    COMMAND_INJECTION_PATTERNS = [
        r"(;\s*|&\s*|&&\s*|\|\s*|\||\|\|\s*|\n\s*|`)(cat|ls|wget|curl|bash|sh|cmd|powershell|python|perl|ruby)",
        r"\$\(.*\)",  # Command substitution
        r"`.*`",      # Backtick substitution
        r">\s*/dev/",
        r"\bexec\b",
        r"\bsystem\b",
        r"\bpopen\b",
    ]
    
    # Path Traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\./",
        r"\.\\.\\",
        r"%2e%2e/",
        r"%2e%2e\\",
        r"\.\.%2f",
        r"\.\.%5c",
        r"/etc/",
        r"/var/",
        r"\\windows\\",
        r"\\winnt\\",
        r"c:\\",
        r"c:/",
    ]
    
    # File Inclusion patterns
    FILE_INCLUSION_PATTERNS = [
        r"(include|require|include_once|require_once)\s*\(",
        r"file\s*=|path\s*=|url\s*=",
        r"(php|jsp|asp|aspx)://",
        r"expect://",
        r"data://",
        r"zip://",
    ]
    
    # LDAP Injection
    LDAP_INJECTION_PATTERNS = [
        r"(\*|&|\||!)\s*\(",
        r"\(uid\s*=",
    ]
    
    # XXE Injection
    XXE_INJECTION_PATTERNS = [
        r"<!ENTITY",
        r"SYSTEM\s*\"",
        r"PUBLIC\s*\"",
    ]
    
    def __init__(self):
        self.compiled_patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> Dict[str, List]:
        """Compile all regex patterns for efficiency"""
        return {
            'sql_injection': [re.compile(p, re.IGNORECASE) for p in self.SQL_INJECTION_PATTERNS],
            'xss': [re.compile(p, re.IGNORECASE) for p in self.XSS_PATTERNS],
            'command_injection': [re.compile(p, re.IGNORECASE) for p in self.COMMAND_INJECTION_PATTERNS],
            'path_traversal': [re.compile(p, re.IGNORECASE) for p in self.PATH_TRAVERSAL_PATTERNS],
            'file_inclusion': [re.compile(p, re.IGNORECASE) for p in self.FILE_INCLUSION_PATTERNS],
            'ldap_injection': [re.compile(p, re.IGNORECASE) for p in self.LDAP_INJECTION_PATTERNS],
            'xxe_injection': [re.compile(p, re.IGNORECASE) for p in self.XXE_INJECTION_PATTERNS],
        }
    
    def detect(self, uri: str, body: str = "", headers: Dict = None) -> Tuple[float, List[str]]:
        """
        Detect attacks using rules
        Returns: (score 0-1, list of detected attack types)
        """
        if headers is None:
            headers = {}
        
        test_text = f"{uri} {body}".lower()
        detected = []
        
        # Check each category
        for category, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(test_text):
                    detected.append(category)
                    break  # Count category once
        
        # Score based on number of categories detected
        score = min(1.0, len(detected) / len(self.compiled_patterns))
        
        return score, detected
    
    def detect_with_details(self, uri: str, body: str = "", headers: Dict = None) -> Dict:
        """Detailed detection with matched patterns"""
        if headers is None:
            headers = {}
        
        details = {
            'is_attack': False,
            'score': 0.0,
            'detected_types': [],
            'matched_patterns': [],
            'recommendation': 'ALLOW'
        }
        
        test_text = f"{uri} {body}"
        
        for category, patterns in self.compiled_patterns.items():
            for i, pattern in enumerate(patterns):
                match = pattern.search(test_text)
                if match:
                    details['detected_types'].append(category)
                    details['matched_patterns'].append({
                        'category': category,
                        'pattern_index': i,
                        'matched_text': match.group()[:50]  # First 50 chars
                    })
        
        if details['detected_types']:
            details['is_attack'] = True
            details['score'] = min(1.0, len(set(details['detected_types'])) / len(self.compiled_patterns))
            details['recommendation'] = 'BLOCK'
        
        return details

# Global rule engine instance
rule_engine = RuleEngine()
