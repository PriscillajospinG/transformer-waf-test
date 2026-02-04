import os
import sys

# Add parent dir to path to import waf modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from waf.utils.log_parser import NginxLogParser
from waf.data.normalizer import RequestNormalizer
from waf.model.tokenizer import HttpTokenizer

LOG_FILE = "../../nginx/logs/access.log"
DATASET_FILE = "dataset.txt"
TOKENIZER_FILE = "../model/tokenizer.json"

def build_dataset():
    parser = NginxLogParser()
    normalizer = RequestNormalizer()
    
    print(f"Reading logs from {LOG_FILE}...")
    
    # Ensure raw output dir
    os.makedirs(os.path.dirname(DATASET_FILE), exist_ok=True)
    
    count = 0
    with open(DATASET_FILE, 'w') as out:
        # Check if log file exists
        if not os.path.exists(LOG_FILE):
             print(f"Log file {LOG_FILE} not found. Try running traffic generation first.")
             return

        for entry in parser.parse_file(LOG_FILE):
            # We filter out our own WAF checks to avoid pollution if they exist in valid logs
            # In our case, WAF checks are internal so might not show up in the same way,
            # but let's filter just in case.
            if "/_waf_check" in entry['uri']:
                continue
                
            normalized = normalizer.normalize(entry)
            out.write(normalized + "\n")
            count += 1
            
    print(f"Processed {count} log entries. Saved to {DATASET_FILE}")
    
    # Train Tokenizer if we have data
    if count > 0:
        print("Training Tokenizer...")
        tokenizer = HttpTokenizer(vocab_size=1000) # Small vocab for test
        tokenizer.train([DATASET_FILE])
        
        os.makedirs(os.path.dirname(TOKENIZER_FILE), exist_ok=True)
        tokenizer.save(TOKENIZER_FILE)
        print(f"Tokenizer saved to {TOKENIZER_FILE}")

if __name__ == "__main__":
    build_dataset()
