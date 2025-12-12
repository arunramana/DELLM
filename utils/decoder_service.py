"""Decoder Service: Converts embeddings to text."""
import torch
import re
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import threading
from utils.config_loader import config

# Global semaphore to limit concurrent GPU usage (only 1 model on GPU at a time)
_gpu_semaphore = threading.Semaphore(1)


class DecoderService:
    """Service to decode embeddings back to text."""
    
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", device: str = "cpu", force_cpu: bool = False):
        """
        Initialize decoder service.
        
        Args:
            model_name: HuggingFace model name or path (should match embedding model)
            device: Device for computation ('cpu' or 'cuda'). Model is loaded on CPU to save GPU memory.
            force_cpu: If True, always use CPU (prevents GPU memory conflicts)
        """
        self.model_name = model_name
        self.device = "cpu" if force_cpu else device
        self.tokenizer = None
        self.model = None
        self.lock = threading.Lock()
        # Load model on CPU to save GPU memory, move to GPU only during processing
        self.model_device = "cpu"
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
            self.model.to(self.model_device)  # Keep model on CPU
            self.model.eval()  # Set to evaluation mode
            print(f"Decoder model loaded successfully on {self.model_device} (computation on {self.device})")
        except Exception as e:
            print(f"Error loading decoder model: {e}")
            raise
    
    def decode(self, embeddings: torch.Tensor, query: str = None, max_length: Optional[int] = None, use_fast_decode: Optional[bool] = None) -> str:
        """
        Generate answer from processed embeddings using the original query.
        
        Strategy (Fast Mode - default):
        1. Pass embeddings through lm_head to get logits
        2. Use argmax to get most likely tokens in parallel
        3. Decode all tokens at once (much faster than autoregressive generation)
        
        Strategy (Slow Mode - fallback):
        1. Use the original query text (if provided) or reconstruct from embeddings
        2. Format it as an instruction prompt
        3. Generate an answer using autoregressive model.generate()
        
        Args:
            embeddings: Processed embeddings tensor of shape [seq_len, hidden_dim]
            query: Original query text (preferred over reconstruction)
            max_length: Maximum new tokens to generate (uses config default if None)
            use_fast_decode: If True, use fast parallel decoding. If False, use slow generation.
                           If None, uses config value (default: True)
        
        Returns:
            Generated answer text
        """
        # Get config values
        if max_length is None:
            max_length = config.get('generation', 'max_tokens', default=256)
        if use_fast_decode is None:
            use_fast_decode = config.get('generation', 'use_fast_decode', default=True)
        math_keywords = config.get('math_detection', 'keywords', default=[])
        math_temp = config.get('generation', 'math_temperature', default=0.3)
        default_temp = config.get('generation', 'default_temperature', default=0.7)
        top_k = config.get('generation', 'top_k', default=10)
        use_chat_format = config.get('prompt_formatting', 'use_chat_format', default=True)
        templates = config.get('prompt_formatting', 'chat_templates', default={})
        
        with self.lock:
            # Determine computation device
            compute_device = self.device if self.device == "cuda" and torch.cuda.is_available() else self.model_device
            
            # Use semaphore to ensure only one model uses GPU at a time
            if compute_device != self.model_device:
                # Acquire GPU semaphore (only 1 model on GPU at a time)
                _gpu_semaphore.acquire()
                try:
                    # Temporarily move model to GPU if needed
                    self.model.to(compute_device)
                    
                    embeddings = embeddings.to(compute_device)
                    
                    with torch.no_grad():
                        if use_fast_decode:
                            # FAST PATH: Direct embedding-to-token conversion (parallel, ~50-100x faster)
                            print(f"[Decoder - Fast] Using fast parallel decoding...")
                            hidden_states = embeddings.unsqueeze(0)  # [1, seq_len, hidden_dim]
                            
                            if not hasattr(self.model, 'lm_head'):
                                raise ValueError(f"Cannot find LM head in model {self.model_name}")
                            
                            # Get logits for all positions at once (single forward pass)
                            logits = self.model.lm_head(hidden_states)  # [1, seq_len, vocab_size]
                            
                            # Get most likely token at each position (parallel)
                            token_ids = torch.argmax(logits, dim=-1)  # [1, seq_len]
                            token_ids_list = token_ids.squeeze(0).cpu().tolist()
                            
                            # Filter out padding tokens
                            token_ids_list = [tid for tid in token_ids_list 
                                            if tid != self.tokenizer.pad_token_id and tid != self.tokenizer.eos_token_id]
                            
                            # Decode all tokens at once
                            answer = self.tokenizer.decode(token_ids_list, skip_special_tokens=True).strip()
                            
                            # Clean up the answer (keep full answer for generation tasks)
                            if answer:
                                # Remove repeated phrases and clean formatting
                                lines = [line.strip() for line in answer.split('\n') if line.strip()]
                                if lines:
                                    # Keep full answer (multiple lines) - don't truncate
                                    answer = '\n'.join(lines)
                            
                            print(f"[Decoder - Fast] Generated answer: {answer[:100]}...")
                        
                        else:
                            # SLOW PATH: Traditional autoregressive generation (kept as fallback)
                            print(f"[Decoder - Slow] Using autoregressive generation...")
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
                            # Detect if this is a math question using config keywords
                            is_math = any(keyword in query_text.lower() for keyword in math_keywords)
                            
                            # Use config-based prompt formatting
                            if use_chat_format:
                                template_key = 'math' if is_math else 'default'
                                template = templates.get(template_key, templates.get('default', '<|user|>\n{query}\n<|assistant|>\n'))
                            else:
                                template_key = 'generic_math' if is_math else 'generic'
                                template = templates.get(template_key, templates.get('generic', 'Answer the following question: {query}\n\nAnswer:'))
                            
                            prompt = template.format(query=query_text)
                            print(f"[Decoder] Prompt: {prompt[:100]}...")
                            
                            # Step 3: Generate answer from the prompt
                            tokenized = self.tokenizer(prompt, return_tensors="pt")
                            input_ids = tokenized.input_ids.to(compute_device)
                            attention_mask = tokenized.attention_mask.to(compute_device)
                            
                            # Generate using the model with config-based temperature
                            gen_temperature = math_temp if is_math else default_temp
                            # Force minimum tokens for non-math queries to ensure complete answers
                            min_tokens = 5 if is_math else min(100, max_length // 2)
                            generated_ids = self.model.generate(
                                input_ids,
                                attention_mask=attention_mask,
                                max_new_tokens=max_length,
                                min_new_tokens=min_tokens,
                                temperature=gen_temperature,
                                top_k=top_k,
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
                    
                    # Move model back to CPU to free GPU memory
                    self.model.to(self.model_device)
                    torch.cuda.empty_cache()
                finally:
                    # Always release semaphore
                    _gpu_semaphore.release()
            else:
                # Process on CPU (no semaphore needed)
                embeddings = embeddings.to(compute_device)
                
                with torch.no_grad():
                    if use_fast_decode:
                        # FAST PATH: Direct embedding-to-token conversion (parallel, ~50-100x faster)
                        print(f"[Decoder - Fast] Using fast parallel decoding...")
                        hidden_states = embeddings.unsqueeze(0)  # [1, seq_len, hidden_dim]
                        
                        if not hasattr(self.model, 'lm_head'):
                            raise ValueError(f"Cannot find LM head in model {self.model_name}")
                        
                        # Get logits for all positions at once (single forward pass)
                        logits = self.model.lm_head(hidden_states)  # [1, seq_len, vocab_size]
                        
                        # Get most likely token at each position (parallel)
                        token_ids = torch.argmax(logits, dim=-1)  # [1, seq_len]
                        token_ids_list = token_ids.squeeze(0).cpu().tolist()
                        
                        # Filter out padding tokens
                        token_ids_list = [tid for tid in token_ids_list 
                                        if tid != self.tokenizer.pad_token_id and tid != self.tokenizer.eos_token_id]
                        
                        # Decode all tokens at once
                        answer = self.tokenizer.decode(token_ids_list, skip_special_tokens=True).strip()
                        
                        # Clean up the answer (keep full answer for generation tasks)
                        if answer:
                            # Remove repeated phrases and clean formatting
                            lines = [line.strip() for line in answer.split('\n') if line.strip()]
                            if lines:
                                # Keep full answer (multiple lines) - don't truncate
                                answer = '\n'.join(lines)
                        
                        print(f"[Decoder - Fast] Generated answer: {answer[:100]}...")
                    
                    else:
                        # SLOW PATH: Traditional autoregressive generation (kept as fallback)
                        print(f"[Decoder - Slow] Using autoregressive generation...")
                        # Step 1: Get the query text
                        if query:
                            query_text = query
                            print(f"[Decoder] Using original query: {query_text[:60]}...")
                        else:
                            print("[Decoder] Reconstructing query from embeddings...")
                            hidden_states = embeddings.unsqueeze(0)
                            
                            if not hasattr(self.model, 'lm_head'):
                                raise ValueError(f"Cannot find LM head in model {self.model_name}")
                            
                            logits = self.model.lm_head(hidden_states)
                            token_ids = torch.argmax(logits, dim=-1)
                            query_token_ids = token_ids.squeeze(0).cpu().tolist()
                            query_token_ids = [tid for tid in query_token_ids if tid != self.tokenizer.pad_token_id]
                            query_text = self.tokenizer.decode(query_token_ids, skip_special_tokens=True).strip()
                            print(f"[Decoder] Reconstructed query: {query_text[:60]}...")
                        
                        # Step 2: Format as instruction prompt
                        is_math = any(keyword in query_text.lower() for keyword in math_keywords)
                        
                        if use_chat_format:
                            template_key = 'math' if is_math else 'default'
                            template = templates.get(template_key, templates.get('default', '<|user|>\n{query}\n<|assistant|>\n'))
                        else:
                            template_key = 'generic_math' if is_math else 'generic'
                            template = templates.get(template_key, templates.get('generic', 'Answer the following question: {query}\n\nAnswer:'))
                        
                        prompt = template.format(query=query_text)
                        print(f"[Decoder] Prompt: {prompt[:100]}...")
                        
                        # Step 3: Generate answer
                        tokenized = self.tokenizer(prompt, return_tensors="pt")
                        input_ids = tokenized.input_ids.to(compute_device)
                        attention_mask = tokenized.attention_mask.to(compute_device)
                        
                        gen_temperature = math_temp if is_math else default_temp
                        # Force minimum tokens for non-math queries to ensure complete answers
                        min_tokens = 5 if is_math else min(100, max_length // 2)
                        generated_ids = self.model.generate(
                            input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=max_length,
                            min_new_tokens=min_tokens,
                            temperature=gen_temperature,
                            top_k=top_k,
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id
                        )
                        
                        generated_tokens = generated_ids[0][input_ids.shape[1]:]
                        answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                        answer = answer.strip()
                        
                        # For math, extract number
                        if is_math:
                            first_line = answer.split('\n')[0].strip()
                            import re
                            numbers = re.findall(r'\b\d+(?:\.\d+)?\b', first_line)
                            if numbers:
                                answer = numbers[0]
                            else:
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

