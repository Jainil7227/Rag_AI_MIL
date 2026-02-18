import math
from typing import List

class SemanticSimilarity:
    """
    Calculator for cosine similarity between vectors.
    """
    
    def __init__(self):
        pass
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        """
        if len(vec1) != len(vec2):
            raise ValueError(f"Vectors must be same length! Got {len(vec1)} and {len(vec2)}")
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def interpret_similarity(self, score: float) -> str:
        """Convert similarity score to human-readable interpretation."""
        if score >= 0.9:
            return "Nearly identical!"
        elif score >= 0.7:
            return "Very similar"
        elif score >= 0.5:
            return "Somewhat similar"
        elif score >= 0.3:
            return "A bit related"
        else:
            return "Quite different"
    
    def compare_multiple(self, base_vec: List[float], compare_vecs: dict) -> dict:
        """
        Compare one vector against multiple others.
        Returns dict sorted by similarity.
        """
        results = {}
        for name, vec in compare_vecs.items():
            results[name] = self.cosine_similarity(base_vec, vec)
            
        return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

if __name__ == "__main__":
    sim = SemanticSimilarity()
    v1 = [1.0, 0.0]
    v2 = [0.0, 1.0]
    print(f"Similarity: {sim.cosine_similarity(v1, v2)}")