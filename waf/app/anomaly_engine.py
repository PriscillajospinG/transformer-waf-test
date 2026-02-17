"""
Anomaly Detection Engine
Zero-day attack detection using embedding similarity and statistical anomalies
"""

import numpy as np
import logging
from typing import List, Tuple, Optional, Dict
from app.config import WAFConfig
from app.utils import cosine_similarity, average_embeddings

logger = logging.getLogger("anomaly-engine")

class AnomalyEngine:
    """Detect zero-day attacks using anomaly detection"""
    
    def __init__(self, cache_size: int = 1000):
        self.cache_size = cache_size
        self.benign_embeddings = []  # Store embeddings from benign requests
        self.benign_centroid = None  # Mean of benign embeddings
    
    def add_benign_embedding(self, embedding: List[float], request_id: str = ""):
        """Add benign request embedding to database"""
        if len(self.benign_embeddings) >= self.cache_size:
            # Remove oldest
            self.benign_embeddings.pop(0)
        
        self.benign_embeddings.append(embedding)
        
        # Recompute centroid
        if self.benign_embeddings:
            self.benign_centroid = average_embeddings(self.benign_embeddings)
    
    def load_benign_embeddings(self, embeddings: List[Tuple[str, List[float]]]):
        """Load benign embeddings from database"""
        self.benign_embeddings = [emb for _, emb in embeddings[-self.cache_size:]]
        if self.benign_embeddings:
            self.benign_centroid = average_embeddings(self.benign_embeddings)
        logger.info(f"Loaded {len(self.benign_embeddings)} benign embeddings")
    
    def detect_anomaly(self, embedding: List[float], threshold: Optional[float] = None) -> Tuple[bool, float, Dict]:
        """
        Detect if a request embedding is an anomaly
        Returns: (is_anomaly, anomaly_score, details)
        """
        if threshold is None:
            threshold = WAFConfig.ANOMALY_SIMILARITY_THRESHOLD
        
        if not self.benign_embeddings or self.benign_centroid is None:
            # Not enough data yet
            return False, 0.0, {'reason': 'insufficient_data'}
        
        details = {}
        
        # Method 1: Distance from centroid
        centroid_distance = 1.0 - cosine_similarity(embedding, self.benign_centroid)
        details['centroid_distance'] = centroid_distance
        
        # Method 2: Closest match distance
        similarities = [cosine_similarity(embedding, benign_emb) for benign_emb in self.benign_embeddings]
        if similarities:
            closest_similarity = max(similarities)
            closest_distance = 1.0 - closest_similarity
            details['closest_distance'] = closest_distance
            details['closest_similarity'] = closest_similarity
        else:
            closest_distance = 1.0
        
        # Method 3: Statistical distance (Mahalanobis-like)
        statistical_distance = self._calculate_statistical_distance(embedding)
        details['statistical_distance'] = statistical_distance
        
        # Combine distances with weights
        weighted_anomaly_score = (
            0.4 * centroid_distance +
            0.4 * closest_distance +
            0.2 * statistical_distance
        )
        
        details['weighted_score'] = weighted_anomaly_score
        details['threshold'] = threshold
        details['is_anomaly'] = weighted_anomaly_score > threshold
        
        return weighted_anomaly_score > threshold, weighted_anomaly_score, details
    
    def _calculate_statistical_distance(self, embedding: List[float]) -> float:
        """
        Calculate statistical distance from benign cluster
        Higher value = more anomalous
        """
        if not self.benign_embeddings:
            return 0.5  # Neutral
        
        # Calculate variance across embeddings
        embedding_array = np.array(self.benign_embeddings)
        std_dev = np.std(embedding_array, axis=0)
        
        # Avoid division by zero
        std_dev = np.where(std_dev < 0.001, 0.001, std_dev)
        
        # Z-score distance
        test_array = np.array(embedding)
        z_scores = np.abs((test_array - np.mean(embedding_array, axis=0)) / std_dev)
        
        # Max z-score indicates how many standard deviations away
        max_z = np.max(z_scores) if len(z_scores) > 0 else 0
        
        # Convert z-score to 0-1 scale
        statistical_distance = min(1.0, max_z / 5.0)  # 5 sigma ≈ 1.0
        
        return statistical_distance
    
    def get_stats(self) -> Dict:
        """Get statistics about benign embedding cache"""
        return {
            'cached_embeddings': len(self.benign_embeddings),
            'cache_capacity': self.cache_size,
            'centroid_computed': self.benign_centroid is not None,
            'cache_fill_percent': (len(self.benign_embeddings) / self.cache_size * 100) if self.cache_size > 0 else 0
        }

class EmbeddingExtractor:
    """Extract embeddings from request features"""
    
    @staticmethod
    def extract_from_transformer(model_output, attention_mask=None) -> List[float]:
        """
        Extract embedding from transformer model output
        Use [CLS] token or mean pooling
        """
        if hasattr(model_output, 'last_hidden_state'):
            hidden = model_output.last_hidden_state
        else:
            hidden = model_output
        
        # Use [CLS] token (first token) or mean pooling
        if hidden.shape[0] > 0:
            cls_token = hidden[0, 0, :].detach().cpu().numpy()
            return cls_token.tolist()
        
        return []
    
    @staticmethod
    def extract_from_features(features: Dict) -> List[float]:
        """Create embedding from request features"""
        embedding = [
            features.get('special_char_count', 0) / 100,  # Normalize
            features.get('sql_keywords', 0) / 10,
            features.get('xss_keywords', 0) / 10,
            features.get('path_traversal_patterns', 0) / 5,
            # Add more features as needed
        ]
        return embedding

# Global anomaly engine instance
anomaly_engine = AnomalyEngine(cache_size=WAFConfig.ANOMALY_EMBEDDING_CACHE_SIZE)
