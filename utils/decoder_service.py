"""Decoder Service: Converts embeddings to text."""
import torch
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
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
            # Ensure pad token is set (some models don't have it by default)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Use AutoModelForCausalLM to get LM head for text generation
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            print(f"Decoder model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading decoder model: {e}")
            raise
    
    def decode(self, embeddings: torch.Tensor, max_length: int = 256) -> str:
        """
        Decode processed embeddings to text using LM head.
        Uses the last hidden state to generate coherent text.
        
        Args:
            embeddings: Processed embeddings tensor of shape [seq_len, hidden_dim]
            max_length: Maximum tokens to generate
        
        Returns:
            Decoded text
        """
        with self.lock:
            embeddings = embeddings.to(self.device)
            
            with torch.no_grad():
                # Strategy: Use the last hidden state (contains most context)
                # and generate text from it using the model's generation
                last_hidden = embeddings[-1:, :].unsqueeze(0)  # [1, 1, hidden_dim]
                
                # Get LM head to convert to logits
                if not hasattr(self.model, 'lm_head'):
                    raise ValueError(f"Cannot find LM head in model {self.model_name}")
                
                logits = self.model.lm_head(last_hidden)  # [1, 1, vocab_size]
                
                # Get the most likely token from the last position
                # Use top-k sampling for better results
                top_k = 10
                top_k_logits, top_k_indices = torch.topk(logits[0, 0, :], top_k)
                
                # Apply softmax with temperature
                temperature = 1.0
                probs = torch.softmax(top_k_logits / temperature, dim=-1)
                
                # Sample from top-k
                sampled_idx = torch.multinomial(probs, 1).item()
                first_token_id = top_k_indices[sampled_idx].item()
                
                # Now use the model to generate continuation from this token
                # Create input_ids starting with the sampled token
                input_ids = torch.tensor([[first_token_id]], device=self.device)
                
                # Generate continuation using the model
                # We'll generate a short sequence
                generated_ids = [first_token_id]
                
                for _ in range(min(max_length - 1, 100)):
                    # Forward pass through model
                    outputs = self.model(input_ids)
                    next_token_logits = outputs.logits[0, -1, :]
                    
                    # Apply top-k filtering
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    probs = torch.softmax(top_k_logits / temperature, dim=-1)
                    next_token_id = top_k_indices[torch.multinomial(probs, 1)].item()
                    
                    generated_ids.append(next_token_id)
                    
                    # Stop on EOS
                    if next_token_id == self.tokenizer.eos_token_id:
                        break
                    
                    # Update input_ids for next iteration
                    input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=self.device)], dim=1)
                
                # Decode generated tokens
                text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                text = text.strip()
            
            return text if text else "Unable to generate text."
    
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

