"""RL Training: Updates node fitness based on rewards."""
from typing import List, Dict, Any, Optional
from core.transformer_node import TransformerNode
from utils.config_loader import config
from utils.embedding_service import EmbeddingService
import re
import torch
import torch.nn.functional as F


class RLTrainer:
    """Reinforcement learning trainer for node fitness updates."""
    
    def __init__(self, embedding_service: Optional[EmbeddingService] = None):
        """
        Initialize RL trainer.
        
        Args:
            embedding_service: Optional embedding service for semantic similarity.
                              If None, falls back to word-overlap method.
        """
        self.embedding_service = embedding_service
    
    def calculate_final_answer_quality(self, final_answer: str, correct_answer: str) -> float:
        """
        Calculate quality score (0.0 to 1.0) for final answer using semantic similarity.
        
        Returns:
            Quality score between 0.0 and 1.0
        """
        if not correct_answer:
            return 0.5  # Default if no correct answer
        
        final_lower = final_answer.lower().strip()
        correct_lower = correct_answer.lower().strip()
        
        # Exact match (fast path)
        if final_lower == correct_lower:
            return 1.0
        
        # Use semantic similarity if embedding service is available
        if self.embedding_service:
            try:
                # Generate embeddings for both answers
                final_emb = self.embedding_service.encode(final_answer)
                correct_emb = self.embedding_service.encode(correct_answer)
                
                # Average embeddings over sequence length to get sentence-level representation
                # Shape: [seq_len, hidden_dim] -> [hidden_dim]
                final_vec = final_emb.mean(dim=0)  # Average pooling
                correct_vec = correct_emb.mean(dim=0)
                
                # Normalize vectors for cosine similarity
                final_vec = F.normalize(final_vec.unsqueeze(0), p=2, dim=1)
                correct_vec = F.normalize(correct_vec.unsqueeze(0), p=2, dim=1)
                
                # Calculate cosine similarity (returns tensor)
                similarity = F.cosine_similarity(final_vec, correct_vec, dim=1)
                semantic_score = similarity.item()  # Convert to Python float
                
                # Also check for number matching (important for math questions)
                final_nums = set(re.findall(r'\b\d+(?:\.\d+)?\b', final_answer))
                correct_nums = set(re.findall(r'\b\d+(?:\.\d+)?\b', correct_answer))
                num_match_ratio = len(final_nums & correct_nums) / len(correct_nums) if correct_nums else 1.0
                
                # Combine semantic similarity (70%) with number matching (30%)
                # This ensures math answers with correct numbers get high scores
                quality = 0.7 * semantic_score + 0.3 * num_match_ratio
                return min(1.0, max(0.0, quality))
                
            except Exception as e:
                # Fallback to word overlap if embedding fails
                print(f"Warning: Semantic similarity failed ({e}), falling back to word overlap")
        
        # Fallback: Word overlap method (if no embedding service or if it fails)
        final_words = set(final_lower.split())
        correct_words = set(correct_lower.split())
        
        if not correct_words:
            return 0.0
        
        overlap = len(final_words & correct_words)
        overlap_ratio = overlap / len(correct_words)
        
        # Number matching
        final_nums = set(re.findall(r'\b\d+(?:\.\d+)?\b', final_answer))
        correct_nums = set(re.findall(r'\b\d+(?:\.\d+)?\b', correct_answer))
        num_match_ratio = len(final_nums & correct_nums) / len(correct_nums) if correct_nums else 1.0
        
        # Combined score
        quality = 0.6 * overlap_ratio + 0.4 * num_match_ratio
        return min(1.0, max(0.0, quality))
    
    def calculate_node_contributions(self, nodes_used: List[TransformerNode], 
                                     assignments: Dict[str, List[TransformerNode]],
                                     aggregated: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate how much each node contributed to final answer.
        
        Returns:
            Dict[node_id, contribution_score] (sums to ~1.0)
        """
        contributions = {}
        total_weight = 0.0
        
        for chunk_id, result in aggregated.items():
            all_responses = result.get('all_responses', [])
            consensus = result.get('consensus', 0.5)
            
            for resp in all_responses:
                node_id = resp.get('node_id')
                if not node_id:
                    continue
                
                # Find node to get fitness
                node = next((n for n in nodes_used if n.node_id == node_id), None)
                if not node:
                    continue
                
                # Contribution = fitness × confidence × consensus
                weight = node.fitness * resp.get('confidence', 0.85) * (0.5 + consensus)
                
                contributions[node_id] = contributions.get(node_id, 0.0) + weight
                total_weight += weight
        
        # Normalize contributions
        if total_weight > 0:
            contributions = {nid: w / total_weight for nid, w in contributions.items()}
        else:
            # Equal distribution if no weights
            num_nodes = len(set(n.node_id for n in nodes_used))
            if num_nodes > 0:
                contributions = {n.node_id: 1.0 / num_nodes for n in nodes_used}
        
        return contributions
    
    def update_fitness(self, nodes_used: List[TransformerNode], 
                      final_answer: str, correct_answer: str,
                      assignments: Dict[str, List[TransformerNode]],
                      aggregated: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Update fitness based on final answer quality.
        
        Args:
            nodes_used: List of nodes that were used
            final_answer: Final assembled answer
            correct_answer: Correct answer for verification
            assignments: Node assignments per chunk
            aggregated: Aggregated results per chunk
        
        Returns:
            Dict of node_id -> new fitness score
        """
        # Calculate final answer quality
        quality_score = self.calculate_final_answer_quality(final_answer, correct_answer)
        
        # Calculate node contributions
        contributions = self.calculate_node_contributions(nodes_used, assignments, aggregated)
        
        # Get config values
        correct_reward = config.get('training', 'correct_reward', default=0.15)
        incorrect_penalty = config.get('training', 'incorrect_penalty', default=-0.10)
        max_reward = config.get('training', 'max_reward', default=0.5)
        min_reward = config.get('training', 'min_reward', default=-0.5)
        alpha = config.get('defaults', 'fitness_update_alpha', default=0.9)
        beta = config.get('defaults', 'fitness_update_beta', default=0.1)
        fast_threshold = config.get('training', 'fast_latency_threshold', default=10.0)
        slow_threshold = config.get('training', 'slow_latency_threshold', default=60.0)
        speed_bonus = config.get('training', 'speed_bonus', default=0.05)
        speed_penalty = config.get('training', 'speed_penalty', default=-0.05)
        
        updates = {}
        
        # Distribute reward based on quality and contribution
        for node in nodes_used:
            node_id = node.node_id
            contribution = contributions.get(node_id, 0.0)
            
            # Get average latency for this node from aggregated results
            node_latency = 1.0
            for chunk_id, result in aggregated.items():
                all_responses = result.get('all_responses', [])
                for resp in all_responses:
                    if resp.get('node_id') == node_id:
                        node_latency = resp.get('latency', 1.0)
                        break
                if node_latency != 1.0:
                    break
            
            # Reward = quality_score * contribution
            # High quality (1.0) with high contribution (1.0) = max reward
            # Low quality (0.0) = penalty
            base_reward = quality_score * correct_reward + (1 - quality_score) * incorrect_penalty
            reward = base_reward * contribution
            
            # Speed bonus (faster = better)
            if node_latency < fast_threshold:
                reward += speed_bonus * contribution
            elif node_latency > slow_threshold:
                reward += speed_penalty * contribution
            
            # Clamp reward
            reward = max(min_reward, min(max_reward, reward))
            
            # Update fitness
            node.fitness = alpha * node.fitness + beta * (node.fitness + reward)
            node.fitness = max(0.0, min(1.0, node.fitness))
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

