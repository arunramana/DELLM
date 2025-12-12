"""Test script for the hybrid query classifier."""
from minimal.query_classifier import HybridClassifier

# Test queries
TEST_QUERIES = [
    # Math operations
    ("what's 10% of 1000", "MATH_OP"),
    ("calculate 5 + 3", "MATH_OP"),
    ("how much is 50 × 2", "MATH_OP"),
    ("solve 100 / 4", "MATH_OP"),
    
    # Factual questions (should NOT be math)
    ("what's the tallest mountain", "FACTUAL_OP"),
    ("who is Albert Einstein", "FACTUAL_OP"),
    ("where is Paris located", "FACTUAL_OP"),
    ("when did World War 2 start", "FACTUAL_OP"),
    ("what is machine learning", "FACTUAL_OP"),
    ("define quantum computing", "FACTUAL_OP"),
    
    # Comparisons
    ("compare Python and Java", "COMPARISON_OP"),
    ("what's the difference between RAM and ROM", "COMPARISON_OP"),
    ("which is better: Mac or Windows", "COMPARISON_OP"),
    
    # Generation
    ("write a story about AI", "GENERATION_OP"),
    ("create a plan for weekend", "GENERATION_OP"),
    
    # Reasoning
    ("explain why the sky is blue", "REASONING_OP"),
    ("how does photosynthesis work", "REASONING_OP"),
]


def test_pattern_only():
    """Test pattern-based classification only."""
    print("="*70)
    print("PATTERN-BASED CLASSIFICATION (Fast, No ML)")
    print("="*70)
    
    classifier = HybridClassifier(use_ml=False)
    
    correct = 0
    total = len(TEST_QUERIES)
    
    for query, expected in TEST_QUERIES:
        result = classifier.classify(query)
        is_correct = result['operation'] == expected
        correct += is_correct
        
        status = "✓" if is_correct else "✗"
        print(f"{status} Query: {query}")
        print(f"  Expected: {expected}")
        print(f"  Got: {result['operation']} (confidence: {result['confidence']:.2f}, method: {result['method']})")
        if not is_correct:
            print(f"  Pattern: {result.get('matched_pattern', 'N/A')}")
        print()
    
    accuracy = (correct / total) * 100
    print(f"Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    print()
    return accuracy


def test_with_ml():
    """Test hybrid classification with ML."""
    print("="*70)
    print("HYBRID CLASSIFICATION (Pattern + ML)")
    print("="*70)
    
    classifier = HybridClassifier(use_ml=True)
    
    if not classifier.use_ml:
        print("⚠️  ML not available, using pattern-only mode")
        print("   Install with: pip install transformers torch")
        print()
        return None
    
    correct = 0
    total = len(TEST_QUERIES)
    
    for query, expected in TEST_QUERIES:
        result = classifier.classify(query, debug=False)
        is_correct = result['operation'] == expected
        correct += is_correct
        
        status = "✓" if is_correct else "✗"
        print(f"{status} Query: {query}")
        print(f"  Expected: {expected}")
        print(f"  Got: {result['operation']} (confidence: {result['confidence']:.2f}, method: {result.get('final_method', 'unknown')})")
        if not is_correct:
            print(f"  Pattern: {result.get('matched_pattern', 'N/A')}")
        print()
    
    accuracy = (correct / total) * 100
    print(f"Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    print()
    return accuracy


def test_specific_case():
    """Test the specific case that was failing."""
    print("="*70)
    print("SPECIFIC TEST: Math vs Factual Distinction")
    print("="*70)
    
    classifier = HybridClassifier(use_ml=False)
    
    test_cases = [
        ("what's 10% of 1000", "MATH_OP"),
        ("what's the tallest mountain", "FACTUAL_OP"),
        ("what is 5 + 5", "MATH_OP"),
        ("what is machine learning", "FACTUAL_OP"),
    ]
    
    for query, expected in test_cases:
        result = classifier.classify(query, debug=False)
        is_correct = result['operation'] == expected
        
        status = "✓" if is_correct else "✗"
        print(f"{status} Query: '{query}'")
        print(f"  Expected: {expected}")
        print(f"  Got: {result['operation']} (confidence: {result['confidence']:.2f})")
        print(f"  Method: {result['method']}, Pattern: {result.get('matched_pattern', 'N/A')}")
        print()


if __name__ == "__main__":
    # Test the specific failing case first
    test_specific_case()
    
    # Test pattern-only (always available)
    pattern_accuracy = test_pattern_only()
    
    # Test with ML (if available)
    ml_accuracy = test_with_ml()
    
    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Pattern-based accuracy: {pattern_accuracy:.1f}%")
    if ml_accuracy is not None:
        print(f"Hybrid (Pattern + ML) accuracy: {ml_accuracy:.1f}%")
        print(f"Improvement: +{ml_accuracy - pattern_accuracy:.1f}%")
    else:
        print("Hybrid mode: Not available (ML dependencies missing)")
    print()
    print("💡 To enable ML classification:")
    print("   pip install transformers torch")
    print()

