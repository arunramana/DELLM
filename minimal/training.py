"""RL Training: Updates node fitness based on rewards."""
from typing import List, Dict, Any
from core.transformer_node import TransformerNode
from utils.config_loader import config


class RLTrainer:
    """Reinforcement learning trainer for node fitness updates."""
    
    def update_fitness(self, nodes_used: List[TransformerNode], aggregated: Dict[str, Dict[str, Any]], 
                      correct_answer: str = None) -> Dict[str, float]:
        """
        Update node fitness based on performance.
        
        Args:
            nodes_used: List of nodes that were used
            aggregated: Aggregated results per chunk
            correct_answer: Optional correct answer for verification
        
        Returns:
            Dict of node_id -> new fitness score
        """
        updates = {}
        
        for chunk_id, result in aggregated.items():
            all_responses = result.get('all_responses', [])
            consensus_answer = result.get('answer', '')
            
            for resp in all_responses:
                node_id = resp.get('node_id')
                node_answer = resp.get('answer', '')
                latency = resp.get('latency', 1.0)
                
                # Get config values
                correct_reward = config.get('training', 'correct_reward', default=0.15)
                incorrect_penalty = config.get('training', 'incorrect_penalty', default=-0.10)
                consensus_correct_reward = config.get('training', 'consensus_correct_reward', default=0.10)
                consensus_incorrect_penalty = config.get('training', 'consensus_incorrect_penalty', default=-0.05)
                fast_threshold = config.get('training', 'fast_latency_threshold', default=10.0)
                slow_threshold = config.get('training', 'slow_latency_threshold', default=60.0)
                speed_bonus = config.get('training', 'speed_bonus', default=0.05)
                speed_penalty = config.get('training', 'speed_penalty', default=-0.05)
                max_reward = config.get('training', 'max_reward', default=0.5)
                min_reward = config.get('training', 'min_reward', default=-0.5)
                alpha = config.get('defaults', 'fitness_update_alpha', default=0.9)
                beta = config.get('defaults', 'fitness_update_beta', default=0.1)
                
                # Calculate reward
                if correct_answer:
                    # Compare with ground truth
                    is_correct = self._check_correctness(node_answer, correct_answer)
                    reward = correct_reward if is_correct else incorrect_penalty
                else:
                    # Use semantic similarity instead of exact match
                    is_correct = self._check_correctness(node_answer, consensus_answer)
                    reward = consensus_correct_reward if is_correct else consensus_incorrect_penalty
                
                # Speed bonus (faster = better)
                if latency < fast_threshold:
                    reward += speed_bonus
                elif latency > slow_threshold:
                    reward += speed_penalty
                
                # Update fitness: reward-based update
                node = next((n for n in nodes_used if n.node_id == node_id), None)
                if node:
                    # Clamp reward to configured range
                    reward = max(min_reward, min(max_reward, reward))
                    # Update: alpha * old + beta * (old + reward)
                    # This way, positive reward increases fitness
                    node.fitness = alpha * node.fitness + beta * (node.fitness + reward)
                    node.fitness = max(0.0, min(1.0, node.fitness))  # Clamp to [0, 1]
                    updates[node_id] = node.fitness
        
        return updates
    
    def _check_correctness(self, answer: str, correct: str) -> bool:
        """Semantic correctness check (improved similarity)."""
        answer_words = set(answer.lower().split())
        correct_words = set(correct.lower().split())
        
        if not correct_words:
            return False
        
        # Word overlap
        overlap = len(answer_words & correct_words)
        overlap_ratio = overlap / len(correct_words)
        
        # Also check for key numbers/entities
        import re
        answer_nums = set(re.findall(r'\d+', answer))
        correct_nums = set(re.findall(r'\d+', correct))
        num_match = len(answer_nums & correct_nums) > 0 if correct_nums else True
        
        # Consider correct if >60% word overlap OR key numbers match
        return overlap_ratio > 0.6 or (overlap_ratio > 0.4 and num_match)

