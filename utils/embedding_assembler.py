"""Embedding Assembler: Progressively assembles embeddings from nodes."""
import torch
from typing import Dict, Optional
from collections import OrderedDict


class EmbeddingAssembler:
    """Assembles embeddings from multiple chunks in correct order."""
    
    def __init__(self):
        """Initialize assembler."""
        self.chunk_embeddings: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.chunk_order: list = []
    
    def add_chunk(self, chunk_id: str, embeddings: torch.Tensor, position: Optional[int] = None):
        """
        Add chunk embeddings in correct position.
        
        Args:
            chunk_id: Chunk identifier (e.g., 'step_1', 'step_2')
            embeddings: Embeddings tensor of shape [seq_len, hidden_dim]
            position: Optional position index (if None, extracted from chunk_id)
        """
        # Extract position from chunk_id if not provided
        if position is None:
            try:
                # Extract number from chunk_id (e.g., 'step_1' -> 1)
                position = int(''.join(filter(str.isdigit, chunk_id)))
            except:
                position = len(self.chunk_embeddings)
        
        # Store with position for sorting
        self.chunk_embeddings[chunk_id] = {
            'embeddings': embeddings,
            'position': position
        }
        self.chunk_order.append(chunk_id)
        print(f"  [Assembler] Added chunk {chunk_id} at position {position}")
    
    def get_assembled(self, include_positions: bool = False) -> torch.Tensor:
        """
        Get assembled embeddings in correct order.
        
        Args:
            include_positions: If True, add positional encoding between chunks
        
        Returns:
            Assembled embeddings tensor of shape [total_seq_len, hidden_dim]
        """
        if not self.chunk_embeddings:
            raise ValueError("No chunks to assemble")
        
        # Sort by position
        sorted_chunks = sorted(
            self.chunk_embeddings.items(),
            key=lambda x: x[1]['position']
        )
        
        # Extract embeddings in order
        embeddings_list = [chunk_data['embeddings'] for _, chunk_data in sorted_chunks]
        
        # Concatenate along sequence dimension
        assembled = torch.cat(embeddings_list, dim=0)  # [total_seq_len, hidden_dim]
        
        print(f"  [Assembler] Assembled {len(embeddings_list)} chunks: {assembled.shape}")
        
        return assembled
    
    def get_chunk_count(self) -> int:
        """Get number of chunks added."""
        return len(self.chunk_embeddings)
    
    def clear(self):
        """Clear all stored chunks."""
        self.chunk_embeddings.clear()
        self.chunk_order.clear()

