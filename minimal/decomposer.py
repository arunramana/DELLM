"""Query Decomposer: Rule-based keyword matching to break queries into chunks."""
import re
from typing import List, Dict, Any
from minimal.query_classifier import get_classifier


class QueryDecomposer:
    """Rule-based query decomposition into operation chunks."""
    
    # Separators that indicate multiple steps
    SEPARATORS = r'\s+(?:then|and|,|\.|after|before|next|also)\s+'
    
    def __init__(self, use_ml_classifier: bool = False):
        """
        Initialize decomposer.
        
        Args:
            use_ml_classifier: Whether to use ML-based classification (requires transformers)
        """
        self.classifier = get_classifier(use_ml=use_ml_classifier)
        self.use_ml = use_ml_classifier
    
    def decompose(self, query: str, target_chunks: int = None) -> List[Dict[str, Any]]:
        """
        Break query into chunks based on keywords and separators.
        If target_chunks is provided, try to split into that many chunks.
        
        Args:
            query: Query string to decompose
            target_chunks: Target number of chunks (will split further if needed)
        
        Returns:
            List of chunks with operation type and text
        """
        query_lower = query.lower()
        
        # First, check if query has natural separators
        parts = re.split(self.SEPARATORS, query, flags=re.IGNORECASE)
        parts = [p.strip() for p in parts if p.strip()]
        
        # If no separators found, try to split by other means
        if len(parts) <= 1:
            # Try to split by sentence boundaries or clauses
            parts = re.split(r'[.!?]\s+|,\s+(?:and|or|but)\s+', query, flags=re.IGNORECASE)
            parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 5]
            
            # If still single part, try to split into semantic chunks
            if len(parts) <= 1 and target_chunks and target_chunks > 1:
                # Split long queries into roughly equal parts
                words = query.split()
                if len(words) > target_chunks * 3:  # At least 3 words per chunk
                    chunk_size = len(words) // target_chunks
                    parts = []
                    for i in range(0, len(words), chunk_size):
                        chunk_words = words[i:i + chunk_size]
                        if chunk_words:
                            parts.append(' '.join(chunk_words))
        
        # If still single part, return as single chunk
        if len(parts) <= 1:
            op = self._detect_operation(query)
            return [{
                'chunk_id': 'step_1',
                'operation': op,
                'text': query,
                'complexity': 0.4 if op == 'MATH_OP' else 0.6
            }]
        
        # If we have fewer parts than target, we have two options:
        # 1. Keep semantic parts intact and assign same chunk to multiple nodes (better)
        # 2. Only split if parts are long enough to split meaningfully
        if target_chunks and len(parts) < target_chunks:
            # Only split if parts are long enough (at least 8 words) and can be split meaningfully
            expanded_parts = []
            
            for part in parts:
                words = part.split()
                # Only split if part is long enough (8+ words) and has natural break points
                if len(words) >= 8:
                    # Try to find natural break points (conjunctions, prepositions)
                    break_points = []
                    for i, word in enumerate(words[1:], 1):  # Skip first word
                        word_lower = word.lower()
                        if word_lower in ['and', 'or', 'with', 'for', 'to', 'in', 'on', 'at']:
                            break_points.append(i)
                    
                    if break_points and len(break_points) >= 1:
                        # Split at natural break points
                        prev = 0
                        for bp in break_points:
                            if prev < bp:
                                sub_part = ' '.join(words[prev:bp])
                                if len(sub_part.strip()) > 5:  # At least 5 chars
                                    expanded_parts.append(sub_part)
                                prev = bp
                        # Add remaining
                        if prev < len(words):
                            sub_part = ' '.join(words[prev:])
                            if len(sub_part.strip()) > 5:
                                expanded_parts.append(sub_part)
                    else:
                        # No natural break points, keep as-is
                        expanded_parts.append(part)
                else:
                    # Part too short to split meaningfully, keep as-is
                    expanded_parts.append(part)
            
            # If we still have fewer chunks than target, that's okay
            # The router will assign the same chunk to multiple nodes
            parts = expanded_parts if expanded_parts else parts
        
        # Create chunks for each part
        chunks = []
        for i, part in enumerate(parts, 1):
            op = self._detect_operation(part)
            chunks.append({
                'chunk_id': f'step_{i}',
                'operation': op,
                'text': part.strip(),
                'complexity': 0.4 if op == 'MATH_OP' else 0.6
            })
        
        return chunks
    
    def _detect_operation(self, text: str) -> str:
        """
        Detect operation type from text using hybrid classifier.
        
        Args:
            text: Query text to classify
            
        Returns:
            Operation type string
        """
        result = self.classifier.classify(text)
        return result['operation']

