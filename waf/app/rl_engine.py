"""
Reinforcement Learning Engine
Q-learning for WAF policy optimization based on feedback
"""

import random
import logging
from typing import Tuple, Dict
from app.config import WAFConfig
from app.utils import extract_state_hash

logger = logging.getLogger("rl-engine")

class RLEngine:
    """Q-Learning based reinforcement learning for WAF"""
    
    def __init__(self):
        self.epsilon = WAFConfig.RL_EPSILON  # Exploration rate
        self.alpha = WAFConfig.RL_ALPHA      # Learning rate
        self.gamma = WAFConfig.RL_GAMMA      # Discount factor
        self.q_table = {}  # state_hash -> {'allow': q_value, 'block': q_value}
    
    def get_action(self, state_hash: str, bert_score: float, rule_score: float, 
                  method: str = "GET", endpoint: str = "/") -> Tuple[str, float]:
        """
        Choose action (allow/block) using epsilon-greedy policy
        Returns: (action, q_value)
        """
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            # Explore: random action
            action = random.choice(['allow', 'block'])
            logger.debug(f"RL: Exploration action={action} for state={state_hash}")
        else:
            # Exploit: best action from Q-table
            action, q_value = self._get_best_action(state_hash, bert_score, rule_score)
            logger.debug(f"RL: Exploitation action={action} q={q_value:.4f} for state={state_hash}")
        
        return action, self._get_q_value(state_hash, action)
    
    def _get_best_action(self, state_hash: str, bert_score: float, rule_score: float) -> Tuple[str, float]:
        """Get action with highest Q-value"""
        allow_q, block_q = self._get_q_values(state_hash)
        
        # Combine Q-values with current detection scores
        # High scores should prefer blocking
        adjusted_allow_q = allow_q * (1 - (bert_score + rule_score) / 2)
        adjusted_block_q = block_q * (bert_score + rule_score) / 2
        
        if adjusted_allow_q > adjusted_block_q:
            return 'allow', adjusted_allow_q
        else:
            return 'block', adjusted_block_q
    
    def _get_q_values(self, state_hash: str) -> Tuple[float, float]:
        """Get Q-values for a state"""
        if state_hash not in self.q_table:
            return 0.0, 0.0
        
        state_data = self.q_table[state_hash]
        return state_data.get('allow', 0.0), state_data.get('block', 0.0)
    
    def _get_q_value(self, state_hash: str, action: str) -> float:
        """Get Q-value for state-action pair"""
        if state_hash not in self.q_table:
            return 0.0
        return self.q_table[state_hash].get(action, 0.0)
    
    def update(self, state_hash: str, action: str, reward: float, next_state_hash: str = None):
        """
        Update Q-values based on reward (feedback)
        Reward scale:
        - +1.0: Correct action (blocked attack, allowed benign)
        - -1.0: Wrong action (blocked benign, allowed attack)
        - 0.0: Uncertain
        """
        if state_hash not in self.q_table:
            self.q_table[state_hash] = {'allow': 0.0, 'block': 0.0}
        
        current_q = self._get_q_value(state_hash, action)
        
        # Get max Q-value from next state
        if next_state_hash:
            _, next_max_q = self._get_q_values(next_state_hash)
        else:
            next_max_q = 0.0
        
        # Q-learning update rule: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s')) - Q(s,a)]
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        
        self.q_table[state_hash][action] = new_q
        
        logger.info(f"RL Update: state={state_hash} action={action} reward={reward} new_q={new_q:.4f}")
    
    def apply_feedback(self, request_id: str, actual_class: str, predicted_action: str,
                      bert_score: float, rule_score: float, method: str, endpoint: str):
        """
        Apply feedback from admin (benign/malicious) and update Q-values
        
        actual_class: 'benign' or 'malicious'
        predicted_action: 'allow' or 'block'
        """
        state_hash = extract_state_hash(method, endpoint, bert_score, rule_score)
        
        # Determine reward
        if actual_class == 'benign' and predicted_action == 'allow':
            reward = 1.0  # Correct: allowed good request
        elif actual_class == 'malicious' and predicted_action == 'block':
            reward = 1.0  # Correct: blocked bad request
        elif actual_class == 'benign' and predicted_action == 'block':
            reward = -1.0  # Wrong: blocked good request (false positive)
        elif actual_class == 'malicious' and predicted_action == 'allow':
            reward = -1.0  # Wrong: allowed bad request (false negative)
        else:
            reward = 0.0
        
        self.update(state_hash, predicted_action, reward)
        logger.info(f"Feedback: request={request_id} decision={actual_class} pred={predicted_action} reward={reward}")
    
    def get_policy_stats(self) -> Dict:
        """Get statistics about current policy"""
        if not self.q_table:
            return {'states': 0, 'avg_q_allow': 0.0, 'avg_q_block': 0.0}
        
        total_allow = sum(v['allow'] for v in self.q_table.values())
        total_block = sum(v['block'] for v in self.q_table.values())
        
        return {
            'states': len(self.q_table),
            'avg_q_allow': total_allow / len(self.q_table),
            'avg_q_block': total_block / len(self.q_table),
            'total_reward': total_allow + total_block
        }

# Global RL engine instance
rl_engine = RLEngine()
