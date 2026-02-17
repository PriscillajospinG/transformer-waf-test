"""
SQLite Database Management for WAF
Stores logs, RL Q-table, embeddings, and admin feedback
"""

import sqlite3
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

DB_PATH = "/app/data/waf.db"
_db_lock = threading.Lock()

def init_db():
    """Initialize all database tables"""
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Requests Log Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT UNIQUE NOT NULL,
            timestamp TEXT NOT NULL,
            ip_address TEXT,
            method TEXT,
            endpoint TEXT,
            full_uri TEXT,
            headers_json TEXT,
            body_preview TEXT,
            bert_score REAL,
            rule_score REAL,
            anomaly_score REAL,
            combined_score REAL,
            prediction TEXT,
            action TEXT,
            blocked INTEGER,
            block_reason TEXT,
            user_feedback TEXT,
            feedback_timestamp TEXT,
            processing_time_ms REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Q-Learning Table (state, action -> Q-value)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS q_table (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            state_hash TEXT UNIQUE NOT NULL,
            state_json TEXT,
            action_allow_q REAL DEFAULT 0.0,
            action_block_q REAL DEFAULT 0.0,
            visit_count INTEGER DEFAULT 0,
            last_update TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Benign Request Embeddings (for zero-day detection)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT UNIQUE NOT NULL,
            embedding_json TEXT NOT NULL,
            uri TEXT,
            method TEXT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Admin Feedback
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT UNIQUE NOT NULL,
            admin_decision TEXT,
            confidence_score REAL,
            notes TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(request_id) REFERENCES requests(request_id)
        )
    ''')
    
    # Statistics/Summary
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hour_key TEXT UNIQUE,
            total_requests INTEGER DEFAULT 0,
            blocked_requests INTEGER DEFAULT 0,
            allowed_requests INTEGER DEFAULT 0,
            average_bert_score REAL DEFAULT 0.0,
            average_rule_score REAL DEFAULT 0.0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def get_connection():
    """Get database connection with thread safety"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Return rows as dicts
    return conn

def log_request(request_id: str, ip: str, method: str, endpoint: str, full_uri: str,
                headers: Dict, body_preview: str, bert_score: float, rule_score: float,
                anomaly_score: float, combined_score: float, prediction: str, action: str,
                blocked: bool, block_reason: str = None, processing_time_ms: float = 0) -> bool:
    """Log request details and detection results"""
    with _db_lock:
        try:
            conn = get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO requests (
                    request_id, timestamp, ip_address, method, endpoint, full_uri,
                    headers_json, body_preview, bert_score, rule_score, anomaly_score,
                    combined_score, prediction, action, blocked, block_reason, processing_time_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                request_id, datetime.utcnow().isoformat(), ip, method, endpoint, full_uri,
                json.dumps(headers), body_preview, bert_score, rule_score, anomaly_score,
                combined_score, prediction, action, int(blocked), block_reason, processing_time_ms
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error logging request: {e}")
            return False

def get_recent_logs(limit: int = 100) -> List[Dict]:
    """Retrieve recent request logs"""
    with _db_lock:
        try:
            conn = get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT request_id, timestamp, ip_address, method, endpoint, blocked,
                       bert_score, rule_score, anomaly_score, block_reason, processing_time_ms
                FROM requests
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            return [dict(row) for row in rows]
        except Exception as e:
            print(f"Error fetching logs: {e}")
            return []

def get_stats() -> Dict:
    """Get aggregated statistics"""
    with _db_lock:
        try:
            conn = get_connection()
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) as total FROM requests')
            total = dict(cursor.fetchone())['total']
            
            cursor.execute('SELECT COUNT(*) as blocked FROM requests WHERE blocked = 1')
            blocked = dict(cursor.fetchone())['blocked']
            
            cursor.execute('SELECT COUNT(*) as allowed FROM requests WHERE blocked = 0')
            allowed = dict(cursor.fetchone())['allowed']
            
            cursor.execute('SELECT AVG(bert_score) as avg_bert FROM requests WHERE bert_score IS NOT NULL')
            avg_bert = dict(cursor.fetchone())['avg_bert'] or 0.0
            
            cursor.execute('SELECT AVG(rule_score) as avg_rule FROM requests WHERE rule_score IS NOT NULL')
            avg_rule = dict(cursor.fetchone())['avg_rule'] or 0.0
            
            cursor.execute('''
                SELECT COUNT(*) as count FROM requests 
                WHERE timestamp >= datetime('now', '-24 hours')
            ''')
            last_24h = dict(cursor.fetchone())['count']
            
            conn.close()
            
            return {
                'total_requests': total,
                'blocked_requests': blocked,
                'allowed_requests': allowed,
                'block_rate': (blocked / total * 100) if total > 0 else 0,
                'average_bert_score': round(avg_bert, 4),
                'average_rule_score': round(avg_rule, 4),
                'requests_last_24h': last_24h
            }
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {}

def save_q_value(state_hash: str, state_json: str, action: str, q_value: float):
    """Save Q-value to table"""
    with _db_lock:
        try:
            conn = get_connection()
            cursor = conn.cursor()
            
            # Try to insert, or update if exists
            cursor.execute('''
                INSERT OR IGNORE INTO q_table (state_hash, state_json, visit_count, last_update)
                VALUES (?, ?, 0, ?)
            ''', (state_hash, state_json, datetime.utcnow().isoformat()))
            
            if action == "allow":
                cursor.execute('''
                    UPDATE q_table SET action_allow_q = ?, visit_count = visit_count + 1, 
                                      last_update = ? WHERE state_hash = ?
                ''', (q_value, datetime.utcnow().isoformat(), state_hash))
            elif action == "block":
                cursor.execute('''
                    UPDATE q_table SET action_block_q = ?, visit_count = visit_count + 1,
                                      last_update = ? WHERE state_hash = ?
                ''', (q_value, datetime.utcnow().isoformat(), state_hash))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error saving Q-value: {e}")

def get_q_values(state_hash: str) -> Tuple[float, float]:
    """Get Q-values for a state (allow_q, block_q)"""
    with _db_lock:
        try:
            conn = get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT action_allow_q, action_block_q FROM q_table WHERE state_hash = ?
            ''', (state_hash,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return float(row[0]), float(row[1])
            return 0.0, 0.0
        except Exception as e:
            print(f"Error getting Q-values: {e}")
            return 0.0, 0.0

def get_q_table(limit: int = 50) -> List[Dict]:
    """Get recent Q-table entries"""
    with _db_lock:
        try:
            conn = get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT state_hash, action_allow_q, action_block_q, visit_count, last_update
                FROM q_table
                ORDER BY visit_count DESC
                LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            return [dict(row) for row in rows]
        except Exception as e:
            print(f"Error fetching Q-table: {e}")
            return []

def save_embedding(request_id: str, embedding: List[float], uri: str, method: str):
    """Save request embedding for zero-day detection"""
    with _db_lock:
        try:
            conn = get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR IGNORE INTO embeddings (request_id, embedding_json, uri, method)
                VALUES (?, ?, ?, ?)
            ''', (request_id, json.dumps(embedding), uri, method))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error saving embedding: {e}")

def get_benign_embeddings(limit: int = 1000) -> List[Tuple[str, List[float]]]:
    """Get benign request embeddings for anomaly detection"""
    with _db_lock:
        try:
            conn = get_connection()
            cursor = conn.cursor()
            
            # Get embeddings from requests marked as benign (allowed + feedbacked as benign)
            cursor.execute('''
                SELECT r.request_id, e.embedding_json
                FROM requests r
                LEFT JOIN embeddings e ON r.request_id = e.request_id
                WHERE r.blocked = 0 AND e.embedding_json IS NOT NULL
                ORDER BY r.timestamp DESC
                LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            result = []
            for row in rows:
                try:
                    embedding = json.loads(row[1])
                    result.append((row[0], embedding))
                except:
                    pass
            return result
        except Exception as e:
            print(f"Error fetching embeddings: {e}")
            return []

def save_feedback(request_id: str, admin_decision: str, confidence: float = 1.0, notes: str = ""):
    """Save admin feedback for request"""
    with _db_lock:
        try:
            conn = get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO feedback (request_id, admin_decision, confidence_score, notes)
                VALUES (?, ?, ?, ?)
            ''', (request_id, admin_decision, confidence, notes))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error saving feedback: {e}")

def get_feedback(request_id: str) -> Optional[Dict]:
    """Get feedback for a request"""
    with _db_lock:
        try:
            conn = get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT request_id, admin_decision, confidence_score, notes, created_at
                FROM feedback WHERE request_id = ?
            ''', (request_id,))
            
            row = cursor.fetchone()
            conn.close()
            return dict(row) if row else None
        except Exception as e:
            print(f"Error getting feedback: {e}")
            return None

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully")
