"""Transformer Node: Processes embeddings through transformer blocks."""
import torch
import time
from typing import Dict, Any, Optional
from transformers import AutoModel
import threading
from torch.optim import AdamW
import torch.nn as nn
from utils.config_loader import config

# Global semaphore to limit concurrent GPU usage (only 1 model on GPU at a time)
_gpu_semaphore = threading.Semaphore(1)


class TransformerNode:
    """Node that processes embeddings through transformer blocks only."""
    
    def __init__(self, node_id: str, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                 device: str = "cpu", num_layers: Optional[int] = None):
        """
        Initialize transformer node.
        
        Args:
            node_id: Unique node identifier
            model_name: HuggingFace model name or path
            device: Device for computation ('cpu' or 'cuda'). Models are loaded on CPU to save GPU memory.
            num_layers: Number of transformer layers to use (None = all layers)
        """
        self.node_id = node_id
        self.model_name = model_name
        self.device = device
        self.num_layers = num_layers
        self.model = None
        self.lock = threading.Lock()
        self.fitness = config.get('defaults', 'initial_fitness', default=0.7)
        self.optimizer = None
        self.training_enabled = False
        # Check config for GPU optimization settings
        self.keep_on_gpu = config.get('training', 'keep_models_on_gpu', default=False) and device == "cuda" and torch.cuda.is_available()
        self.use_mixed_precision = config.get('training', 'use_mixed_precision', default=True) and device == "cuda"
        self.use_torch_compile = config.get('training', 'use_torch_compile', default=False) and device == "cuda"
        
        # Load model on CPU by default, but can keep on GPU if configured
        self.model_device = device if self.keep_on_gpu else "cpu"
        self._load_model()
    
    def _load_model(self):
        """Load model and extract transformer blocks."""
        try:
            # Always load on CPU first to avoid OOM during initialization
            load_kwargs = {}
            if self.keep_on_gpu and self.use_mixed_precision:
                # Load with torch_dtype=float16 for faster loading and less memory
                load_kwargs['dtype'] = torch.float16
            
            self.model = AutoModel.from_pretrained(self.model_name, **load_kwargs)
            self.model.eval()  # Set to evaluation mode
            
            # Move to GPU only if keep_on_gpu is enabled, and use semaphore to serialize
            if self.keep_on_gpu and self.device == "cuda" and torch.cuda.is_available():
                _gpu_semaphore.acquire()
                try:
                    # Clear GPU cache before moving model
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                    # Check available GPU memory before moving
                    if torch.cuda.is_available():
                        # Estimate model size (rough approximation: num_params * bytes_per_param)
                        # For FP16, it's 2 bytes per parameter
                        total_params = sum(p.numel() for p in self.model.parameters())
                        estimated_size_gb = (total_params * 2) / (1024 ** 3)  # Convert to GB
                        
                        # Get available memory using mem_get_info (returns free, total)
                        free_memory_bytes, total_memory_bytes = torch.cuda.mem_get_info(0)
                        free_memory_gb = free_memory_bytes / (1024 ** 3)
                        
                        # Only move to GPU if we have enough space (leave 1GB buffer for processing)
                        if free_memory_gb > (estimated_size_gb + 1.0):
                            self.model.to(self.device)
                            self.model_device = self.device
                            torch.cuda.empty_cache()
                        else:
                            # Not enough space, keep on CPU
                            print(f"[{self.node_id}] Warning: Not enough GPU memory ({free_memory_gb:.2f}GB available, need ~{estimated_size_gb:.2f}GB). Keeping model on CPU.")
                            self.model_device = "cpu"
                            self.keep_on_gpu = False  # Disable keep_on_gpu for this node
                finally:
                    _gpu_semaphore.release()
            else:
                # Keep on CPU
                self.model_device = "cpu"
            
            # Detect model architecture and get transformer layers (must be before torch.compile)
            # Check if it's a Llama model by class name
            model_type = type(self.model).__name__.lower()
            self.is_llama = 'llama' in model_type or hasattr(self.model, 'model') and hasattr(self.model.model, 'layers')
            
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                # Llama with nested model structure
                self.transformer_layers = self.model.model.layers
            elif hasattr(self.model, 'layers'):
                # LlamaModel or similar (layers directly on model)
                self.transformer_layers = self.model.layers
                if 'llama' in model_type:
                    self.is_llama = True
            elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                self.transformer_layers = self.model.transformer.h
            else:
                raise ValueError(f"Could not find transformer layers in model {self.model_name}")
            
            # Note: For Llama models, we cannot limit layers using num_layers
            # because we need to use the full model forward method
            if self.num_layers and not self.is_llama:
                self.transformer_layers = self.transformer_layers[:self.num_layers]
            
            # Optional: Compile model for faster inference (PyTorch 2.0+)
            if self.use_torch_compile and hasattr(torch, 'compile'):
                try:
                    if self.is_llama:
                        if hasattr(self.model, 'model'):
                            self.model.model = torch.compile(self.model.model, mode='reduce-overhead')
                        else:
                            self.model = torch.compile(self.model, mode='reduce-overhead')
                    print(f"[{self.node_id}] Model compiled with torch.compile()")
                except Exception as e:
                    print(f"[{self.node_id}] Warning: torch.compile() failed: {e}")
        except Exception as e:
            print(f"Error loading transformer node {self.node_id}: {e}")
            raise
    
    def process_embeddings(self, embeddings: torch.Tensor) -> Dict[str, Any]:
        """
        Process embeddings through transformer blocks.
        
        Args:
            embeddings: Input embeddings tensor of shape [seq_len, hidden_dim]
        
        Returns:
            Dict with processed embeddings and metadata
        """
        start_time = time.time()
        
        with self.lock:
            # Add batch dimension: [seq_len, hidden] -> [1, seq_len, hidden]
            # Move embeddings to computation device
            hidden_states = embeddings.unsqueeze(0).to(self.device)
            
            # If model is on CPU but we want GPU computation, temporarily move to GPU
            # Otherwise, process on the model's current device
            compute_device = self.device if self.device == "cuda" and torch.cuda.is_available() else self.model_device
            
            # Always use semaphore for GPU operations to prevent OOM (only 1 model on GPU at a time)
            if compute_device == "cuda" and torch.cuda.is_available():
                # Acquire GPU semaphore (only 1 model on GPU at a time)
                _gpu_semaphore.acquire()
                try:
                    # Move model to GPU if not already there
                    if self.model_device != compute_device:
                        self.model.to(compute_device)
                    
                    # Generate position_ids for Llama models (needed for positional embeddings)
                    seq_len = hidden_states.shape[1]
                    position_ids = torch.arange(0, seq_len, dtype=torch.long, device=compute_device).unsqueeze(0)
                    
                    # Use mixed precision for faster inference (2x speedup)
                    autocast_context = torch.autocast(device_type='cuda', dtype=torch.float16) if self.use_mixed_precision and compute_device == "cuda" else torch.no_grad()
                    with autocast_context, torch.no_grad():
                        # For Llama models, always use model forward method
                        if self.is_llama:
                            # Use the model's forward method which handles positional embeddings correctly
                            # Check if model has nested structure or is direct LlamaModel
                            if hasattr(self.model, 'model'):
                                llama_model = self.model.model
                            else:
                                llama_model = self.model
                            
                            outputs = llama_model(
                                inputs_embeds=hidden_states,
                                position_ids=position_ids,
                                output_hidden_states=True,
                                return_dict=True
                            )
                            # Get the last hidden state (after all transformer layers)
                            hidden_states = outputs.last_hidden_state
                        else:
                            # For non-Llama architectures, process layers directly
                            # Generate position_ids if needed (some models may need it)
                            for layer in self.transformer_layers:
                                try:
                                    # Try with position_ids first
                                    layer_output = layer(hidden_states, position_ids=position_ids)
                                except TypeError:
                                    # If position_ids not supported, try without
                                    layer_output = layer(hidden_states)
                                
                                if isinstance(layer_output, tuple):
                                    hidden_states = layer_output[0]
                                else:
                                    hidden_states = layer_output
                    
                    # Move model back to CPU to free GPU memory (unless keeping on GPU)
                    if not self.keep_on_gpu:
                        self.model.to(self.model_device)
                        # Aggressive cleanup for sequential processing with multiple nodes
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()  # Wait for all GPU operations to finish
                        torch.cuda.reset_peak_memory_stats()  # Reset memory tracking
                        import gc
                        gc.collect()  # Force Python garbage collection
                finally:
                    # Always release semaphore
                    _gpu_semaphore.release()
            else:
                # Process on CPU (no semaphore needed)
                # Generate position_ids for Llama models
                seq_len = hidden_states.shape[1]
                position_ids = torch.arange(0, seq_len, dtype=torch.long, device=self.model_device).unsqueeze(0)
                
                # Mixed precision not useful on CPU
                with torch.no_grad():
                    # Use model's forward method to properly handle positional embeddings
                    if self.is_llama:
                        if hasattr(self.model, 'model'):
                            llama_model = self.model.model
                        else:
                            llama_model = self.model
                        
                        outputs = llama_model(
                            inputs_embeds=hidden_states,
                            position_ids=position_ids,
                            output_hidden_states=True,
                            return_dict=True
                        )
                        hidden_states = outputs.last_hidden_state
                    else:
                        # For non-Llama architectures, process layers directly
                        for layer in self.transformer_layers:
                            try:
                                layer_output = layer(hidden_states, position_ids=position_ids)
                            except TypeError:
                                layer_output = layer(hidden_states)
                            
                            if isinstance(layer_output, tuple):
                                hidden_states = layer_output[0]
                            else:
                                hidden_states = layer_output
            
            # Remove batch dimension: [1, seq_len, hidden] -> [seq_len, hidden]
            processed_embeddings = hidden_states.squeeze(0)
        
        latency = time.time() - start_time
        
        # Default confidence from config
        confidence = config.get('defaults', 'default_confidence', default=0.85)
        
        return {
            "embeddings": processed_embeddings,
            "confidence": confidence,
            "latency": latency,
            "node_id": self.node_id,
            "input_shape": list(embeddings.shape),
            "output_shape": list(processed_embeddings.shape)
        }
    
    def enable_training(self, learning_rate: float = 1e-5):
        """Enable training mode and initialize optimizer."""
        self.training_enabled = True
        # Only optimize transformer layers
        self.optimizer = AdamW(self.transformer_layers.parameters(), lr=learning_rate)
    
    def train_step(self, input_embeddings: torch.Tensor, target_embeddings: torch.Tensor, 
                   quality_weight: float = 1.0) -> float:
        """
        Train on single example (online learning).
        
        Args:
            input_embeddings: Input embeddings [seq_len, hidden_dim]
            target_embeddings: Target embeddings [seq_len, hidden_dim]
            quality_weight: Weight based on answer quality (0.0 to 1.0)
        
        Returns:
            Loss value
        """
        if not self.training_enabled or self.optimizer is None:
            return 0.0
        
        with self.lock:
            self.model.train()  # Switch to training mode
            
            # Determine computation device
            compute_device = self.device if self.device == "cuda" and torch.cuda.is_available() else self.model_device
            
            # Always use semaphore for GPU operations to prevent OOM (only 1 model on GPU at a time)
            if compute_device == "cuda" and torch.cuda.is_available():
                # Acquire GPU semaphore (only 1 model on GPU at a time)
                _gpu_semaphore.acquire()
                try:
                    # Move model to GPU if not already there
                    if self.model_device != compute_device:
                        self.model.to(compute_device)
                        # Recreate optimizer with parameters on new device
                        if self.optimizer is not None:
                            self.optimizer = AdamW(self.transformer_layers.parameters(), lr=self.optimizer.param_groups[0]['lr'])
                    
                    self.model.train()  # Switch to training mode
                    
                    # Use mixed precision for training (faster, less memory)
                    autocast_context = torch.autocast(device_type='cuda', dtype=torch.float16) if self.use_mixed_precision and compute_device == "cuda" else torch.enable_grad()
                    with autocast_context:
                        # Forward pass
                        hidden_states = input_embeddings.unsqueeze(0).to(compute_device)
                        # Generate position_ids for Llama models
                        seq_len = hidden_states.shape[1]
                        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=compute_device).unsqueeze(0)
                        
                        # Use model's forward method to properly handle positional embeddings
                        if self.is_llama:
                            if hasattr(self.model, 'model'):
                                llama_model = self.model.model
                            else:
                                llama_model = self.model
                            
                            outputs = llama_model(
                                inputs_embeds=hidden_states,
                                position_ids=position_ids,
                                output_hidden_states=True,
                                return_dict=True
                            )
                            hidden_states = outputs.last_hidden_state
                        else:
                            # For non-Llama architectures, process layers directly
                            for layer in self.transformer_layers:
                                try:
                                    layer_output = layer(hidden_states, position_ids=position_ids)
                                except TypeError:
                                    layer_output = layer(hidden_states)
                                
                                if isinstance(layer_output, tuple):
                                    hidden_states = layer_output[0]
                                else:
                                    hidden_states = layer_output
                        
                        output_embeddings = hidden_states.squeeze(0)
                        
                        # Loss: MSE weighted by quality
                        target_emb = target_embeddings.to(compute_device)
                        # Handle dimension mismatch if needed
                        if output_embeddings.shape != target_emb.shape:
                            # Truncate or pad to match
                            min_len = min(output_embeddings.shape[0], target_emb.shape[0])
                            output_embeddings = output_embeddings[:min_len, :]
                            target_emb = target_emb[:min_len, :]
                        
                        loss = nn.functional.mse_loss(output_embeddings, target_emb)
                        weighted_loss = loss * quality_weight
                    
                    # Backward pass (with gradient scaling for mixed precision)
                    self.optimizer.zero_grad()
                    if self.use_mixed_precision and compute_device == "cuda":
                        # Use gradient scaling for mixed precision training
                        try:
                            from torch.amp import GradScaler
                            if not hasattr(self, 'scaler'):
                                self.scaler = GradScaler('cuda')
                        except ImportError:
                            # Fallback for older PyTorch versions
                            from torch.cuda.amp import GradScaler
                            if not hasattr(self, 'scaler'):
                                self.scaler = GradScaler()
                        self.scaler.scale(weighted_loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        weighted_loss.backward()
                        self.optimizer.step()
                    
                    self.model.eval()  # Switch back to eval mode
                    
                    # Move model back to CPU to free GPU memory (unless keeping on GPU)
                    if not self.keep_on_gpu:
                        self.model.to(self.model_device)
                        # Aggressive cleanup for training with multiple nodes
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()  # Wait for all GPU operations to finish
                        import gc
                        gc.collect()  # Force Python garbage collection
                finally:
                    # Always release semaphore
                    _gpu_semaphore.release()
            else:
                # Process on CPU (no semaphore needed)
                with torch.enable_grad():
                    hidden_states = input_embeddings.unsqueeze(0).to(compute_device)
                    # Generate position_ids for Llama models
                    seq_len = hidden_states.shape[1]
                    position_ids = torch.arange(0, seq_len, dtype=torch.long, device=compute_device).unsqueeze(0)
                    
                    # Use model's forward method to properly handle positional embeddings
                    if self.is_llama:
                        if hasattr(self.model, 'model'):
                            llama_model = self.model.model
                        else:
                            llama_model = self.model
                        
                        outputs = llama_model(
                            inputs_embeds=hidden_states,
                            position_ids=position_ids,
                            output_hidden_states=True,
                            return_dict=True
                        )
                        hidden_states = outputs.last_hidden_state
                    else:
                        # For non-Llama architectures, process layers directly
                        for layer in self.transformer_layers:
                            try:
                                layer_output = layer(hidden_states, position_ids=position_ids)
                            except TypeError:
                                layer_output = layer(hidden_states)
                            
                            if isinstance(layer_output, tuple):
                                hidden_states = layer_output[0]
                            else:
                                hidden_states = layer_output
                    
                    output_embeddings = hidden_states.squeeze(0)
                    
                    # Loss: MSE weighted by quality
                    target_emb = target_embeddings.to(compute_device)
                    if output_embeddings.shape != target_emb.shape:
                        min_len = min(output_embeddings.shape[0], target_emb.shape[0])
                        output_embeddings = output_embeddings[:min_len, :]
                        target_emb = target_emb[:min_len, :]
                    
                    loss = nn.functional.mse_loss(output_embeddings, target_emb)
                    weighted_loss = loss * quality_weight
                    
                    # Backward pass
                    self.optimizer.zero_grad()
                    weighted_loss.backward()
                    self.optimizer.step()
                
                self.model.eval()  # Switch back to eval mode
            
            return weighted_loss.item()
    
    def update_fitness(self, fitness: float):
        """Update node fitness score."""
        self.fitness = max(0.0, min(1.0, fitness))
    
    def close(self):
        """Close model in this node."""
        if self.model:
            del self.model
            self.model = None
        if self.optimizer:
            del self.optimizer
            self.optimizer = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

