import numpy as np
import model
from src.physics import (
    compute_word_phase_vector,
    phase_similarity,
    FieldEngine
)
from src.semantics.arabic_semantics import CharacterSemanticEmbedding

class TestCleanMirnanAlHuroofPhysics:
    def test_phase_vector_shape(self):
        v = compute_word_phase_vector("علم")
        assert v.shape == (22,), f"Base phase vector shape should be (22,), got {v.shape}"
        assert np.linalg.norm(v) > 0, "Phase vector should not be zero vector"

    def test_semantic_vector_shape(self):
        sem = CharacterSemanticEmbedding()
        v = sem.get_word_semantic_vector("علم")
        assert v.shape == (29,), f"Semantic phase vector shape should be (29,), got {v.shape}"
        assert np.linalg.norm(v) > 0, "Semantic vector should not be zero vector"

    def test_similarity_bounds(self):
        v1 = compute_word_phase_vector("عدل")
        v2 = compute_word_phase_vector("ظلم")
        sim = phase_similarity(v1, v2)
        assert -1.0 <= sim <= 1.0, f"Similarity should be in range [-1, 1], got {sim}"

    def test_field_engine_results(self):
        vocab = model.load_vocab()
        engine = FieldEngine(vocab)
        res = engine.find_attraction_repulsion("عدل", space_type="combined", top_k=5)
        
        assert "attracted" in res and "repelled" in res
        assert len(res["attracted"]) > 0, "Should return attracted words"
        assert len(res["repelled"]) > 0, "Should return repelled words"
        
        # Verify score descending order for attracted
        att_scores = [score for _, score in res["attracted"]]
        assert att_scores == sorted(att_scores, reverse=True), "Attracted scores should be descending"

        # Verify score ascending order for repelled
        rep_scores = [score for _, score in res["repelled"]]
        assert rep_scores == sorted(rep_scores), "Repelled scores should be ascending"

    def test_geometric_field_engine_results(self):
        vocab = model.load_vocab()
        engine = FieldEngine(vocab)
        res = engine.find_attraction_repulsion("عدل", space_type="geometric", top_k=5)
        
        assert "attracted" in res and "repelled" in res
        assert len(res["attracted"]) > 0, "Should return attracted words"
        assert len(res["repelled"]) > 0, "Should return repelled words"
        
        # Verify score descending order for attracted
        att_scores = [score for _, score in res["attracted"]]
        assert att_scores == sorted(att_scores, reverse=True), "Attracted scores should be descending"

        # Verify score ascending order for repelled
        rep_scores = [score for _, score in res["repelled"]]
        assert rep_scores == sorted(rep_scores), "Repelled scores should be ascending"

