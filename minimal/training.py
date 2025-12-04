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
                    reward = 0.05 if is_correct else -0.05
                else:
                    # Use consensus (if node agrees with consensus, reward)
                    is_correct = (node_answer == consensus_answer)
                    reward = 0.05 if is_correct else -0.02
                
                # Speed bonus (faster = better)
                if latency < 1.0:
                    reward += 0.01
                elif latency > 3.0:
                    reward -= 0.01
                
                # Update fitness: moving average
                node = next((n for n in nodes_used if n.node_id == node_id), None)
                if node:
                    node.fitness = 0.9 * node.fitness + 0.1 * (0.5 + reward)
                    node.fitness = max(0.0, min(1.0, node.fitness))  # Clamp
                    updates[node_id] = node.fitness
        
        return updates
    
    def _check_correctness(self, answer: str, correct: str) -> bool:
        """Simple correctness check (word overlap)."""
        answer_words = set(answer.lower().split())
        correct_words = set(correct.lower().split())
        overlap = len(answer_words & correct_words)
        return overlap / len(correct_words) > 0.7 if correct_words else False

