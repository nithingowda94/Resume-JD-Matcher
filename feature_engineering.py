"""
Phase 4: Feature Engineering Module.

Constructs the final numeric feature vector for Resume-Job matching:
[tfidf_similarity, skill_match_score, experience_match_score, semantic_similarity, resume_skill_count, job_skill_count]
"""

import math
from typing import List, Tuple, Optional, Any
import numpy as np

# 1. Scikit-learn for TF-IDF & Cosine Similarity
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# 2. Sentence Transformers for Semantic Embeddings (Optional)
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    _embedding_model: Optional[SentenceTransformer] = None
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    _embedding_model = None


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> Optional[Any]:
    global _embedding_model
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return None

    if _embedding_model is None:
        try:
            _embedding_model = SentenceTransformer(model_name)
        except Exception:
            return None

    return _embedding_model


def compute_tfidf_similarity(text1: str, text2: str) -> float:
    """Compute Cosine Similarity between texts using Scikit-Learn TfidfVectorizer."""
    if not text1.strip() or not text2.strip():
        return 0.0

    if SKLEARN_AVAILABLE:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            sim_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            return round(max(0.0, min(1.0, float(sim_matrix[0][0]))), 4)
        except Exception:
            pass

    # Pure NumPy Fallback
    words1, words2 = text1.lower().split(), text2.lower().split()
    vocab = sorted(list(set(words1 + words2)))
    if not vocab:
        return 0.0

    v1 = np.array([words1.count(w) for w in vocab], dtype=float)
    v2 = np.array([words2.count(w) for w in vocab], dtype=float)
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return round(float(np.dot(v1, v2) / (norm1 * norm2)), 4)


def compute_semantic_similarity(text1: str, text2: str, model_name: str = "all-MiniLM-L6-v2") -> float:
    """Compute Semantic Embedding Cosine Similarity using Sentence Transformers."""
    if not text1.strip() or not text2.strip():
        return 0.0

    model = get_embedding_model(model_name)
    if model is not None:
        try:
            embeddings = model.encode([text1, text2], convert_to_numpy=True)
            emb1, emb2 = embeddings[0], embeddings[1]
            norm1, norm2 = np.linalg.norm(emb1), np.linalg.norm(emb2)
            if norm1 > 0 and norm2 > 0:
                cosine_sim = float(np.dot(emb1, emb2) / (norm1 * norm2))
                return round(max(0.0, min(1.0, cosine_sim)), 4)
        except Exception:
            pass

    # Graceful semantic fallback (Jaccard token overlap estimate)
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    if not union:
        return 0.0
    jaccard = len(intersection) / len(union)
    semantic_estimate = min(1.0, jaccard * 1.5 + 0.35) if intersection else 0.0
    return round(float(semantic_estimate), 4)


def build_numeric_feature_vector(
    tfidf_sim: float,
    skill_match: float,
    exp_match: float,
    semantic_sim: float,
    resume_skill_cnt: int,
    job_skill_cnt: int
) -> Tuple[np.ndarray, List[float]]:
    """
    Combine all matching metrics into the exact 6-element numeric feature vector:
    [tfidf_similarity, skill_match_score, experience_match_score, semantic_similarity, resume_skill_count, job_skill_count]
    """
    def sanitize(val: float, min_val: float = 0.0, max_val: Optional[float] = None) -> float:
        if val is None or math.isnan(val) or math.isinf(val):
            return 0.0
        val = max(min_val, val)
        if max_val is not None:
            val = min(max_val, val)
        return float(val)

    vector_list = [
        round(sanitize(tfidf_sim, 0.0, 1.0), 4),
        round(sanitize(skill_match, 0.0, 1.0), 4),
        round(sanitize(exp_match, 0.0, 1.0), 4),
        round(sanitize(semantic_sim, 0.0, 1.0), 4),
        int(max(0, resume_skill_cnt)),
        int(max(0, job_skill_cnt))
    ]

    return np.array(vector_list, dtype=float), vector_list
