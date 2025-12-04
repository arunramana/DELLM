"""Decoder Service: Converts embeddings to text."""
import torch
from typing import Optional
from transformers import AutoModel, AutoTokenizer
import threading


class DecoderService:
    """Service to decode embeddings back to text."""
    
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", device: str = "cpu"):
        """
        Initialize decoder service.
        
        Args:
            model_name: HuggingFace model name or path (should match embedding model)
            device: Device to run on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        self.lock = threading.Lock()
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer."""
        try:
            print(f"Loading decoder model: {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            print(f"Decoder model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading decoder model: {e}")
            raise
    
    def decode(self, embeddings: torch.Tensor, max_length: int = 256) -> str:
        """
        Decode embeddings to text by finding closest tokens.
        
        Args:
            embeddings: Embeddings tensor of shape [seq_len, hidden_dim]
            max_length: Maximum tokens to decode (not used, kept for compatibility)
        
        Returns:
            Decoded text
        """
        with self.lock:
            embeddings = embeddings.to(self.device)
            
            with torch.no_grad():
                # Get input embeddings from model (vocabulary embeddings)
                input_embeddings = self.model.get_input_embeddings().weight  # [vocab_size, hidden_dim]
                
                # Normalize for cosine similarity
                embeddings_norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # [seq_len, hidden_dim]
                input_norm = torch.nn.functional.normalize(input_embeddings, p=2, dim=1)  # [vocab_size, hidden_dim]
                
                # Compute cosine similarity: [seq_len, vocab_size]
                similarities = torch.matmul(embeddings_norm, input_norm.t())
                
                # Get token IDs with highest similarity
                token_ids = torch.argmax(similarities, dim=1)  # [seq_len]
                
                # Decode tokens to text (convert to list and remove special tokens)
                token_ids_list = token_ids.cpu().tolist()
                text = self.tokenizer.decode(token_ids_list, skip_special_tokens=True)
            
            return text
    
    def decode_with_generation(self, embeddings: torch.Tensor, max_new_tokens: int = 256) -> str:
        """
        Decode embeddings and generate continuation.
        
        Args:
            embeddings: Embeddings tensor
            max_new_tokens: Maximum new tokens to generate
        
        Returns:
            Generated text
        """
        with self.lock:
            # This is a more complex approach - would need to pass embeddings through
            # transformer layers and then generate. For now, use simple decode.
            return self.decode(embeddings, max_new_tokens)
    
    def close(self):
        """Cleanup model."""
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

