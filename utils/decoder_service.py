"""Decoder Service: Converts embeddings to text."""
import torch
import re
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
    
    def decode(self, embeddings: torch.Tensor, query: str = None, max_length: int = 256) -> str:
        """
        Generate answer from processed embeddings using the original query.
        
        Strategy:
        1. Use the original query text (if provided) or reconstruct from embeddings
        2. Format it as an instruction prompt
        3. Generate an answer using the model
        
        Args:
            embeddings: Processed embeddings tensor of shape [seq_len, hidden_dim]
            query: Original query text (preferred over reconstruction)
            max_length: Maximum new tokens to generate
        
        Returns:
            Generated answer text
        """
        with self.lock:
            embeddings = embeddings.to(self.device)
            
            with torch.no_grad():
                # Step 1: Get the query text
                if query:
                    # Use the original query (preferred)
                    query_text = query
                    print(f"[Decoder] Using original query: {query_text[:60]}...")
                else:
                    # Fallback: Try to reconstruct from embeddings (may not work well)
                    print("[Decoder] Reconstructing query from embeddings...")
                    hidden_states = embeddings.unsqueeze(0)
                    
                    if not hasattr(self.model, 'lm_head'):
                        raise ValueError(f"Cannot find LM head in model {self.model_name}")
                    
                    logits = self.model.lm_head(hidden_states)  # [1, seq_len, vocab_size]
                    token_ids = torch.argmax(logits, dim=-1)  # [1, seq_len]
                    query_token_ids = token_ids.squeeze(0).cpu().tolist()
                    query_token_ids = [tid for tid in query_token_ids if tid != self.tokenizer.pad_token_id]
                    query_text = self.tokenizer.decode(query_token_ids, skip_special_tokens=True).strip()
                    print(f"[Decoder] Reconstructed query: {query_text[:60]}...")
                
                # Step 2: Format as instruction prompt for answer generation
                # Detect if this is a math question and use a more direct format
                is_math = any(keyword in query_text.lower() for keyword in ['%', 'percent', 'calculate', 'what is', 'what\'s', '+', '-', '*', '/', '='])
                
                if "TinyLlama" in self.model_name or "tinyllama" in self.model_name.lower():
                    # TinyLlama chat format
                    if is_math:
                        # For math, ask for just the number
                        prompt = f"<|user|>\n{query_text}\nAnswer with just the number:\n<|assistant|>\n"
                    else:
                        prompt = f"<|user|>\n{query_text}\n<|assistant|>\n"
                else:
                    # Generic instruction format
                    if is_math:
                        prompt = f"Question: {query_text}\nAnswer (just the number):"
                    else:
                        prompt = f"Answer the following question: {query_text}\n\nAnswer:"
                
                print(f"[Decoder] Prompt: {prompt[:100]}...")
                
                # Step 3: Generate answer from the prompt
                input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
                
                # Generate using the model
                # Use lower temperature for math to get more deterministic results
                gen_temperature = 0.3 if is_math else 0.7
                generated_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_length,
                    temperature=gen_temperature,
                    top_k=10,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                # Extract only the generated part (remove the prompt)
                generated_tokens = generated_ids[0][input_ids.shape[1]:]
                
                # Decode generated tokens to text
                answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                answer = answer.strip()
                
                # For math, try to extract just the number from the first line
                if is_math:
                    # Take first line only
                    first_line = answer.split('\n')[0].strip()
                    # Extract number if present
                    import re
                    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', first_line)
                    if numbers:
                        answer = numbers[0]
                    else:
                        # Try pattern like "= 100"
                        match = re.search(r'=\s*(\d+(?:\.\d+)?)', first_line)
                        if match:
                            answer = match.group(1)
                
                print(f"[Decoder] Generated answer: {answer[:100]}...")
            
            return answer if answer else "Unable to generate answer."
    
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

