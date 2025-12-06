"""Configuration Loader: Loads settings from config files."""
import json
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """Loads and provides access to configuration settings."""
    
    _instance = None
    _settings = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._settings is None:
            self._load_settings()
    
    def _load_settings(self):
        """Load settings from config file."""
        config_path = Path(__file__).parent.parent / "config" / "settings.json"
        try:
            with open(config_path, 'r') as f:
                self._settings = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Config file not found at {config_path}, using defaults")
            self._settings = self._get_defaults()
        except json.JSONDecodeError as e:
            print(f"Error parsing config file: {e}, using defaults")
            self._settings = self._get_defaults()
    
    def _get_defaults(self) -> Dict[str, Any]:
        """Get default settings if config file is missing."""
        return {
            "defaults": {
                "initial_fitness": 0.7,
                "default_confidence": 0.85,
                "fitness_update_alpha": 0.9,
                "fitness_update_beta": 0.1
            },
            "generation": {
                "math_temperature": 0.3,
                "default_temperature": 0.7,
                "top_k": 10,
                "max_tokens": 256,
                "chunk_max_tokens": 128
            },
            "timeouts": {
                "node_timeout_seconds": 120.0,
                "client_timeout_seconds": 300.0
            },
            "training": {
                "correct_reward": 0.15,
                "incorrect_penalty": -0.10,
                "consensus_correct_reward": 0.10,
                "consensus_incorrect_penalty": -0.05,
                "fast_latency_threshold": 10.0,
                "slow_latency_threshold": 60.0,
                "speed_bonus": 0.05,
                "speed_penalty": -0.05,
                "max_reward": 0.5,
                "min_reward": -0.5
            },
            "answer_processing": {
                "max_answer_length": 200,
                "min_line_length": 10
            },
            "math_detection": {
                "keywords": ["%", "percent", "calculate", "what is", "what's", "+", "-", "*", "/", "="],
                "enable_calculation": True,
                "calculation_tolerance": 0.01
            },
            "web_search": {
                "enabled": True,
                "factual_keywords": [
                    "tallest", "highest", "largest", "smallest", "longest", "shortest",
                    "first", "last", "oldest", "newest", "famous", "known for",
                    "located in", "capital of", "president of", "population of",
                    "when did", "who is", "what is", "where is"
                ],
                "max_results": 2
            },
            "prompt_formatting": {
                "use_chat_format": True,
                "chat_templates": {
                    "default": "<|user|>\n{query}\n<|assistant|>\n",
                    "math": "<|user|>\n{query}\nAnswer with just the number:\n<|assistant|>\n",
                    "generic": "Answer the following question: {query}\n\nAnswer:",
                    "generic_math": "Question: {query}\nAnswer (just the number):"
                }
            },
            "router": {
                "latency_weight": 0.7,
                "fitness_weight": 0.3
            }
        }
    
    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Get config value by nested keys.
        
        Args:
            *keys: Nested keys (e.g., 'generation', 'temperature')
            default: Default value if key not found
        
        Returns:
            Config value or default
        """
        value = self._settings
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        return value if value is not None else default
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire config section."""
        return self._settings.get(section, {})
    
    def reload(self):
        """Reload settings from file."""
        self._load_settings()


# Global instance
config = ConfigLoader()

