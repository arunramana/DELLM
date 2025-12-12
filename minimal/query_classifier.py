"""Query Classifier: Robust hybrid classification for query types."""
import re
from typing import Dict, Any, Optional
from enum import Enum


class OperationType(Enum):
    """Operation types with clear definitions."""
    MATH_OP = "MATH_OP"
    FACTUAL_OP = "FACTUAL_OP"
    COMPARISON_OP = "COMPARISON_OP"
    GENERATION_OP = "GENERATION_OP"
    REASONING_OP = "REASONING_OP"


class PatternClassifier:
    """
    Pattern-based query classification using regex and priority rules.
    
    Priority order (highest to lowest):
    1. MATH_OP - Has numbers + operators
    2. COMPARISON_OP - Explicit comparison keywords
    3. GENERATION_OP - Creative/production keywords
    4. FACTUAL_OP - Question words + factual keywords
    5. REASONING_OP - Complex/multi-step indicators
    """
    
    # Pattern-based rules (pattern, priority_score)
    PATTERNS = {
        'MATH_OP': [
            # Must have numbers AND operators
            (r'\d+\s*[\+\-\*\/×÷]\s*\d+', 100),  # "5 + 3"
            (r'\d+\s*%\s+of\s+\d+', 100),        # "10% of 100"
            (r'(calculate|compute|solve)\s+\d+', 90),  # "calculate 50"
            (r'what[\'\s]?s?\s+\d+.*[\+\-\*\/×÷%]', 85),  # "what's 5+5"
            (r'how much is\s+\d+', 80),          # "how much is 50"
        ],
        
        'FACTUAL_OP': [
            # Facts, definitions, specific info
            (r'(who is|who was|who\'s)\s+\w+', 95),  # "who is Einstein"
            (r'(where is|where\'s)\s+\w+', 95),      # "where is Paris"
            (r'(what is|what\'s)\s+(?!.*[\d\+\-\*\/])', 85),  # "what is ML" (no math)
            (r'(when did|when was)', 90),            # "when did WWII start"
            (r'\b(tallest|highest|largest|smallest|longest|shortest|oldest|newest)\b', 90),  # Superlatives
            (r'(capital of|president of|population of)', 90),  # Specific facts
            (r'(define|definition|meaning of)', 85),  # Definitions
        ],
        
        'COMPARISON_OP': [
            # Comparisons and analysis
            (r'\b(compare|contrast|versus|vs\.?)\b', 95),
            (r'difference between', 90),
            (r'(better|worse|faster|slower)\s+(than|compared)', 90),
            (r'(pros and cons|advantages and disadvantages)', 95),
            (r'(similar|different|alike)', 75),
        ],
        
        'GENERATION_OP': [
            # Creative/production tasks
            (r'(write|create|generate|compose|draft)\s+(a|an|the)', 95),
            (r'(plan|design|build|make)\s+(a|an|the)', 90),
            (r'give me (ideas|suggestions|examples)', 85),
        ],
        
        'REASONING_OP': [
            # Complex reasoning
            (r'(explain|describe|analyze)\s+(?:why|how)', 85),
            (r'(if .* then|suppose|assume)', 90),
            (r'(step by step|walk through)', 95),
        ]
    }
    
    # Fallback keyword matching (lower priority)
    KEYWORDS = {
        'MATH_OP': ['calculate', 'compute', 'solve'],
        'FACTUAL_OP': ['define', 'definition', 'meaning', 'fact', 'tell me about'],
        'COMPARISON_OP': ['which', 'best', 'worst'],
        'GENERATION_OP': ['write', 'create', 'generate'],
        'REASONING_OP': ['why', 'how', 'explain', 'reason']
    }
    
    def classify(self, text: str) -> Dict[str, Any]:
        """
        Classify query operation type using patterns.
        
        Args:
            text: Query text to classify
            
        Returns:
            Dict with 'operation', 'confidence', and 'matched_pattern'
        """
        text_lower = text.lower().strip()
        
        best_match = {
            'operation': 'FACTUAL_OP',  # Default
            'confidence': 0.5,
            'matched_pattern': None,
            'method': 'default'
        }
        
        # Check pattern-based rules (priority order)
        for op_type, patterns in self.PATTERNS.items():
            for pattern, priority in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    confidence = priority / 100.0
                    if confidence > best_match['confidence']:
                        best_match = {
                            'operation': op_type,
                            'confidence': confidence,
                            'matched_pattern': pattern,
                            'method': 'pattern'
                        }
        
        # If no strong pattern match, use keyword fallback
        if best_match['confidence'] < 0.7:
            for op_type, keywords in self.KEYWORDS.items():
                if any(kw in text_lower for kw in keywords):
                    if best_match['confidence'] < 0.65:
                        best_match = {
                            'operation': op_type,
                            'confidence': 0.65,
                            'matched_pattern': 'keyword_fallback',
                            'method': 'keyword'
                        }
        
        return best_match
    
    def is_math_operation(self, text: str) -> bool:
        """Quick check if text is a math operation."""
        result = self.classify(text)
        return result['operation'] == 'MATH_OP' and result['confidence'] > 0.75


class MLClassifier:
    """ML-based query classifier using zero-shot classification (optional)."""
    
    def __init__(self):
        """Initialize ML classifier."""
        self.classifier = None
        self.available = False
        self._init_model()
    
    def _init_model(self):
        """Initialize the ML model (lazy loading)."""
        try:
            from transformers import pipeline
            import warnings
            warnings.filterwarnings("ignore", category=FutureWarning)
            
            print("[Classifier] Loading ML classification model...")
            # Use a small, fast model
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1  # CPU only for classifier
            )
            
            self.labels = [
                "mathematical calculation or arithmetic",
                "factual question or definition",
                "comparison or analysis",
                "creative writing or generation",
                "logical reasoning or explanation"
            ]
            
            self.label_to_op = {
                "mathematical calculation or arithmetic": "MATH_OP",
                "factual question or definition": "FACTUAL_OP",
                "comparison or analysis": "COMPARISON_OP",
                "creative writing or generation": "GENERATION_OP",
                "logical reasoning or explanation": "REASONING_OP"
            }
            
            self.available = True
            print("[Classifier] ML model loaded successfully")
            
        except ImportError:
            print("[Classifier] ML model not available (transformers not installed or no model)")
            print("[Classifier] Install with: pip install transformers torch")
            self.available = False
        except Exception as e:
            print(f"[Classifier] Could not load ML model: {e}")
            self.available = False
    
    def classify(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Classify using ML model.
        
        Args:
            text: Query text to classify
            
        Returns:
            Dict with classification results or None if unavailable
        """
        if not self.available or not self.classifier:
            return None
        
        try:
            result = self.classifier(text, self.labels, multi_label=False)
            top_label = result['labels'][0]
            confidence = result['scores'][0]
            
            return {
                'operation': self.label_to_op[top_label],
                'confidence': confidence,
                'matched_pattern': f"ml:{top_label}",
                'method': 'ml',
                'all_scores': dict(zip(
                    [self.label_to_op[l] for l in result['labels']],
                    result['scores']
                ))
            }
        except Exception as e:
            print(f"[Classifier] ML classification error: {e}")
            return None


class HybridClassifier:
    """
    Hybrid classifier combining rule-based and ML-based approaches.
    
    Strategy:
    1. Always use rule-based classification (fast, reliable)
    2. If confidence is high (>0.85), use rule result
    3. If confidence is low (<0.70) and ML available, use ML
    4. Otherwise use rule result
    """
    
    def __init__(self, use_ml: bool = False):
        """
        Initialize hybrid classifier.
        
        Args:
            use_ml: Whether to enable ML-based classification (requires transformers)
        """
        self.pattern_classifier = PatternClassifier()
        self.ml_classifier = MLClassifier() if use_ml else None
        self.use_ml = use_ml and (self.ml_classifier is not None and self.ml_classifier.available)
        
        if use_ml and not self.use_ml:
            print("[Classifier] ML requested but not available, using pattern-only mode")
    
    def classify(self, text: str, debug: bool = False) -> Dict[str, Any]:
        """
        Hybrid classification combining rules and ML.
        
        Args:
            text: Query text to classify
            debug: If True, return detailed classification info
            
        Returns:
            Dict with 'operation', 'confidence', and metadata
        """
        # Always get rule-based result (fast baseline)
        rule_result = self.pattern_classifier.classify(text)
        
        if debug:
            print(f"[Classifier] Rule-based: {rule_result['operation']} "
                  f"(confidence: {rule_result['confidence']:.2f})")
        
        # If high confidence from rules, use it
        if rule_result['confidence'] > 0.85:
            rule_result['final_method'] = 'pattern_high_confidence'
            return rule_result
        
        # If ML available and rules uncertain, use ML
        if self.use_ml and rule_result['confidence'] < 0.70:
            ml_result = self.ml_classifier.classify(text)
            
            if ml_result and ml_result['confidence'] > 0.75:
                if debug:
                    print(f"[Classifier] ML override: {ml_result['operation']} "
                          f"(confidence: {ml_result['confidence']:.2f})")
                ml_result['final_method'] = 'ml_override'
                ml_result['rule_result'] = rule_result
                return ml_result
        
        # Otherwise return rule-based result
        rule_result['final_method'] = 'pattern_default'
        return rule_result
    
    def is_math_operation(self, text: str) -> bool:
        """Quick check if text is a math operation."""
        result = self.classify(text)
        return result['operation'] == 'MATH_OP' and result['confidence'] > 0.70


# Global classifier instance (singleton)
_classifier_instance = None


def get_classifier(use_ml: bool = False) -> HybridClassifier:
    """
    Get or create the global classifier instance.
    
    Args:
        use_ml: Whether to enable ML classification
        
    Returns:
        HybridClassifier instance
    """
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = HybridClassifier(use_ml=use_ml)
    return _classifier_instance


# Convenience functions
def classify_query(text: str, use_ml: bool = False) -> Dict[str, Any]:
    """Classify a query using the hybrid classifier."""
    classifier = get_classifier(use_ml=use_ml)
    return classifier.classify(text)


def is_math_query(text: str, use_ml: bool = False) -> bool:
    """Check if query is a math operation."""
    classifier = get_classifier(use_ml=use_ml)
    return classifier.is_math_operation(text)

