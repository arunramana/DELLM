"""Simple LLM client wrapper for local models using llama-cpp-python."""
import json
import time
import os
import threading
from typing import Dict, Any, Optional

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("Warning: llama-cpp-python not installed. Using mock mode.")

# Thread locks per model file to prevent concurrent access
_model_locks: Dict[str, threading.Lock] = {}
_locks_lock = threading.Lock()  # Lock for accessing _model_locks dict

# Track all loaded models for cleanup
_loaded_models: Dict[str, 'LLMClient'] = {}
_models_lock = threading.Lock()  # Lock for accessing _loaded_models dict


class LLMClient:
    """LLM client supporting local models via llama-cpp-python or mock responses."""
    
    def __init__(self, model_path: Optional[str] = None, model_name: str = "mock", 
                 api_key: Optional[str] = None, n_ctx: int = 2048, n_threads: int = 4,
                 n_gpu_layers: int = -1):
        """
        Initialize LLM client.
        
        Args:
            model_path: Path to GGUF model file (.gguf)
            model_name: Model name (for mock mode or fallback)
            api_key: Not used for local models, kept for compatibility
            n_ctx: Context window size
            n_threads: Number of threads for inference
            n_gpu_layers: Number of layers to offload to GPU (-1 = all layers, 0 = CPU only)
        """
        self.model_path = model_path
        self.model_name = model_name
        self.n_ctx = n_ctx
        # Reduce threads when used in parallel to avoid conflicts
        self.n_threads = 1 if n_threads > 1 else n_threads  # Use 1 thread for thread safety
        self.n_gpu_layers = n_gpu_layers
        self.llm = None
        self.model_type = "chat"  # Assume chat models by default
        
        # Get or create lock for this model file
        self.model_lock = None
        if model_path:
            with _locks_lock:
                if model_path not in _model_locks:
                    _model_locks[model_path] = threading.Lock()
                self.model_lock = _model_locks[model_path]
        
        # Determine mode
        self.use_local = model_path is not None and os.path.exists(model_path) and LLAMA_CPP_AVAILABLE
        self.use_mock = not self.use_local
        
        # Load model if path provided and exists
        if self.use_local:
            try:
                print(f"Loading model from {model_path}...")
                # Use chat_format if available (for llama-cpp-python >= 0.2.0)
                # This handles chat formatting automatically
                try:
                    from llama_cpp import llama_chat_format
                    # Try to use chat format handler if available
                    self.llm = Llama(
                        model_path=model_path,
                        n_ctx=n_ctx,
                        n_threads=self.n_threads,  # Use reduced thread count
                        n_gpu_layers=self.n_gpu_layers,  # GPU acceleration
                        verbose=False,
                        chat_format="llama-3" if "llama-3" in model_path.lower() else None
                    )
                except (ImportError, TypeError):
                    # Fallback to basic Llama if chat_format not available
                    self.llm = Llama(
                        model_path=model_path,
                        n_ctx=n_ctx,
                        n_threads=self.n_threads,  # Use reduced thread count
                        n_gpu_layers=self.n_gpu_layers,  # GPU acceleration
                        verbose=False
                    )
                print(f"Model loaded successfully: {model_path}")
                # Track loaded model for cleanup
                with _models_lock:
                    _loaded_models[model_path] = self
            except Exception as e:
                print(f"Warning: Failed to load model {model_path}: {e}")
                print("Falling back to mock mode")
                self.use_local = False
                self.use_mock = True
    
    def close(self):
        """Close and cleanup the model."""
        if self.llm and self.use_local:
            try:
                # llama-cpp-python models don't have explicit close, but we can delete the reference
                # The model will be garbage collected
                if hasattr(self.llm, '__del__'):
                    # Try to call destructor if available
                    try:
                        self.llm.__del__()
                    except:
                        pass
                self.llm = None
                self.use_local = False
                self.use_mock = True
                
                # Remove from tracking
                if self.model_path:
                    with _models_lock:
                        _loaded_models.pop(self.model_path, None)
                
                print(f"Model closed: {self.model_path}")
            except Exception as e:
                print(f"Error closing model {self.model_path}: {e}")
    
    def __del__(self):
        """Destructor - cleanup on deletion."""
        if self.llm:
            try:
                self.close()
            except:
                pass
    
    def complete(self, prompt: str, temperature: float = 0.3, max_tokens: int = 512) -> str:
        """
        Complete a prompt using local model or mock.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
        """
        if self.use_local and self.llm:
            return self._local_complete(prompt, temperature, max_tokens)
        else:
            return self._mock_complete(prompt)
    
    def _format_chat_prompt(self, query: str) -> str:
        """Format query as a chat prompt for instruction-tuned models."""
        # Try different chat formats based on model type
        model_path_lower = (self.model_path or "").lower()
        
        # TinyLlama chat format (don't add begin_of_text, model handles it)
        if "tinyllama" in model_path_lower:
            return f"<|user|>\n{query}\n<|assistant|>\n"
        
        # Llama 3.x format (don't add begin_of_text, model handles it)
        if "llama-3" in model_path_lower or "llama3" in model_path_lower:
            # llama-cpp-python automatically adds <|begin_of_text|>, so we don't include it
            return f"<|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        # Generic instruction format
        if "instruct" in model_path_lower or "chat" in model_path_lower:
            return f"### Instruction:\n{query}\n\n### Response:\n"
        
        # Default: return as-is (let the model handle formatting)
        return query
    
    def _truncate_prompt(self, prompt: str, max_chars: int = None) -> str:
        """Truncate prompt to fit within context window."""
        if max_chars is None:
            # Estimate: n_ctx tokens * ~4 chars per token, leave room for response
            max_chars = int(self.n_ctx * 3)  # Conservative estimate
        
        if len(prompt) <= max_chars:
            return prompt
        
        # Truncate from the middle, keeping start and end
        start_len = max_chars // 2
        end_len = max_chars - start_len - 10  # Leave room for truncation marker
        truncated = prompt[:start_len] + "\n[...truncated...]\n" + prompt[-end_len:]
        print(f"Warning: Prompt truncated from {len(prompt)} to {len(truncated)} chars (context: {self.n_ctx})")
        return truncated
    
    def _local_complete(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Call local llama-cpp-python model with thread safety."""
        if not self.llm:
            return self._mock_complete(prompt)
        
        # Acquire lock for this model file to prevent concurrent access
        lock = self.model_lock if self.model_lock else threading.Lock()
        
        try:
            with lock:
                # Truncate prompt to fit in context window
                prompt = self._truncate_prompt(prompt)
                
                # Try using create_chat_completion if available (better for chat models)
                model_path_lower = (self.model_path or "").lower()
                use_chat_api = "llama-3" in model_path_lower or "llama3" in model_path_lower or "chat" in model_path_lower
                
                if use_chat_api and hasattr(self.llm, 'create_chat_completion'):
                    try:
                        # Use chat completion API which handles formatting automatically
                        response = self.llm.create_chat_completion(
                            messages=[
                                {"role": "user", "content": prompt}
                            ],
                            temperature=temperature,
                            max_tokens=min(max_tokens, self.n_ctx // 2),  # Ensure max_tokens fits in context
                            stop=["<|user|>", "<|assistant|>", "<|eot_id|>"]
                        )
                        result_text = response['choices'][0]['message']['content'].strip()
                        if result_text and len(result_text) >= 3:
                            return result_text
                    except Exception as e:
                        # Fall back to regular completion if chat API fails
                        if "tensor" in str(e).lower() or "bounds" in str(e).lower():
                            print(f"Tensor bounds error in chat API, trying shorter prompt...")
                            # Try with even shorter prompt
                            shorter_prompt = self._truncate_prompt(prompt, max_chars=int(self.n_ctx * 2))
                            try:
                                response = self.llm.create_chat_completion(
                                    messages=[{"role": "user", "content": shorter_prompt}],
                                    temperature=temperature,
                                    max_tokens=min(max_tokens, self.n_ctx // 4),
                                    stop=["<|user|>", "<|assistant|>", "<|eot_id|>"]
                                )
                                result_text = response['choices'][0]['message']['content'].strip()
                                if result_text and len(result_text) >= 3:
                                    return result_text
                            except:
                                pass
                
                # Format prompt for chat models (without duplicate tokens)
                formatted_prompt = self._format_chat_prompt(prompt)
                
                # Try with formatted prompt
                # Ensure max_tokens doesn't exceed available context
                safe_max_tokens = min(max_tokens, self.n_ctx // 2)
                
                try:
                    response = self.llm(
                        formatted_prompt,
                        temperature=temperature,
                        max_tokens=safe_max_tokens,
                        stop=["<|user|>", "<|assistant|>", "<|eot_id|>", "\n\n\n", "### Instruction:", "Human:", "User:"],
                        echo=False
                    )
                except Exception as e:
                    if "tensor" in str(e).lower() or "bounds" in str(e).lower():
                        print(f"Tensor bounds error, trying with shorter prompt and fewer tokens...")
                        # Try with even shorter prompt and fewer tokens
                        shorter_prompt = self._truncate_prompt(formatted_prompt, max_chars=int(self.n_ctx * 2))
                        response = self.llm(
                            shorter_prompt,
                            temperature=temperature,
                            max_tokens=min(256, self.n_ctx // 4),
                            stop=["<|user|>", "<|assistant|>", "<|eot_id|>"],
                            echo=False
                        )
                    else:
                        raise
                
                result_text = response['choices'][0]['text'].strip()
                
                # If result is empty or just punctuation, try different approaches
                if not result_text or len(result_text) < 3 or result_text in [".", "!", "?", "..."]:
                    # Try 1: Without chat formatting, just the raw prompt
                    try:
                        response = self.llm(
                            prompt,
                            temperature=temperature,
                            max_tokens=safe_max_tokens,
                            stop=["\n\n\n"],
                            echo=False
                        )
                        result_text = response['choices'][0]['text'].strip()
                    except:
                        pass
                
                # If still empty, try with higher temperature (but don't increase tokens to avoid bounds error)
                if not result_text or len(result_text) < 3:
                    try:
                        response = self.llm(
                            formatted_prompt,
                            temperature=min(temperature + 0.3, 0.9),
                            max_tokens=min(safe_max_tokens, 512),  # Don't exceed safe limit
                            stop=["<|user|>", "<|assistant|>"],
                            echo=False
                        )
                        result_text = response['choices'][0]['text'].strip()
                    except Exception as e:
                        if "tensor" in str(e).lower() or "bounds" in str(e).lower():
                            print(f"Tensor bounds error in retry, skipping...")
                        pass
                
                # Final check and warning
                if not result_text or len(result_text) < 3:
                    print(f"Warning: Model '{self.model_path}' returned minimal output: '{result_text}'")
                    print(f"  Prompt preview: {prompt[:100]}...")
                    # Return a helpful message instead of empty
                    return f"[Model generated minimal output. The model may need different prompt formatting or the query may be too complex for this model size.]"
                
                return result_text
        except Exception as e:
            error_msg = str(e).lower()
            if "access violation" in error_msg or "assert" in error_msg or "bounds" in error_msg:
                print(f"Model access error with {self.model_path}: {e}")
                print("  This may be due to concurrent access. Falling back to mock mode for this request.")
                return self._mock_complete(prompt)
            else:
                print(f"Error during inference with model {self.model_path}: {e}")
                import traceback
                traceback.print_exc()
                return self._mock_complete(prompt)
    
    def _mock_complete(self, prompt: str) -> str:
        """Mock LLM responses for testing."""
        prompt_lower = prompt.lower()
        
        # Simulate latency
        time.sleep(0.1)
        
        # Mock simplicity scoring
        if "score this query's simplicity" in prompt_lower:
            if "what is" in prompt_lower and ("+" in prompt_lower or "×" in prompt_lower or "*" in prompt_lower):
                return '{"score": 0.95, "reason": "Simple arithmetic"}'
            elif "capital" in prompt_lower:
                return '{"score": 0.80, "reason": "Simple factual question"}'
            elif "plan" in prompt_lower and "trip" in prompt_lower:
                return '{"score": 0.35, "reason": "Complex multi-step planning"}'
            else:
                return '{"score": 0.60, "reason": "Medium complexity"}'
        
        # Mock decomposition
        if "decompose this query" in prompt_lower:
            if "trip" in prompt_lower:
                return json.dumps([
                    {"query": "Find hotels in Boston under $150/night", "simplicity": 0.65},
                    {"query": "Create a 2-day itinerary for Boston", "simplicity": 0.55},
                    {"query": "Calculate total trip budget", "simplicity": 0.75}
                ])
            else:
                return json.dumps([
                    {"query": prompt.split("\n")[0], "simplicity": 0.70}
                ])
        
        # Mock synthesis
        if "synthesize" in prompt_lower:
            return "Here is a synthesized answer based on the provided results."
        
        # Mock direct answers
        if "what is" in prompt_lower:
            if "2+2" in prompt_lower or "2 + 2" in prompt_lower:
                return "4"
            elif "15+27" in prompt_lower or "15 + 27" in prompt_lower:
                return "42"
            elif "25×17" in prompt_lower or "25 * 17" in prompt_lower:
                return "425"
        
        # Default mock response
        return "This is a mock response. Replace with real LLM calls."


def cleanup_all_models():
    """Close all loaded models."""
    with _models_lock:
        models_to_close = list(_loaded_models.values())
        _loaded_models.clear()
    
    for client in models_to_close:
        try:
            client.close()
        except Exception as e:
            print(f"Error closing model {client.model_path}: {e}")
    
    print(f"Closed {len(models_to_close)} model(s)")


def get_loaded_models() -> list:
    """Get list of currently loaded model paths."""
    with _models_lock:
        return list(_loaded_models.keys())
