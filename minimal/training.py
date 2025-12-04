"""RL Training: Updates node fitness based on rewards."""
from typing import List, Dict, Any
from core.node import Node


class RLTrainer:
    """Reinforcement learning trainer for node fitness updates."""
    
    def update_fitness(self, nodes_used: List[Node], aggregated: Dict[str, Dict[str, Any]], 
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
                
                # Calculate reward
                if correct_answer:
                    # Compare with ground truth
                    is_correct = self._check_correctness(node_answer, correct_answer)
                    reward = 0.15 if is_correct else -0.10  # Increased rewards
                else:
                    # Use semantic similarity instead of exact match
                    is_correct = self._check_correctness(node_answer, consensus_answer)
                    reward = 0.10 if is_correct else -0.05  # Increased rewards
                
                # Speed bonus (faster = better, adjusted for local models)
                if latency < 10.0:  # Adjusted threshold for local models
                    reward += 0.05
                elif latency > 60.0:
                    reward -= 0.05
                
                # Update fitness: reward-based update (fixes the formula)
                node = next((n for n in nodes_used if n.node_id == node_id), None)
                if node:
                    # New formula: fitness increases when reward > 0
                    # Clamp reward to [-0.5, 0.5] range
                    reward = max(-0.5, min(0.5, reward))
                    # Update: 0.9 * old + 0.1 * (old + reward)
                    # This way, positive reward increases fitness
                    node.fitness = 0.9 * node.fitness + 0.1 * (node.fitness + reward)
                    node.fitness = max(0.0, min(1.0, node.fitness))  # Clamp
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

