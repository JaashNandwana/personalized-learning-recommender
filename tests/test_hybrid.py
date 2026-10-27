"""
Unit tests for Hybrid Recommendation Engine.
"""

import pytest
from jaash_rs.config import RecommenderConfig
from jaash_rs.domain.resources import CATALOG
from jaash_rs.models.bkt import BayesianKnowledgeTracer
from jaash_rs.models.collaborative import BiasedMatrixFactorizationALS
from jaash_rs.models.content_filtering import ContentBasedRecommender
from jaash_rs.models.hybrid_engine import HybridRecommendationEngine
from jaash_rs.profile.student_profile import StudentProfile


def test_hybrid_recommendation_and_shelves():
    config = RecommenderConfig(alpha=0.4, beta=0.3, gamma=0.3, weakness_threshold_theta=0.6)
    bkt = BayesianKnowledgeTracer(config)
    cb = ContentBasedRecommender(CATALOG[:40])
    cf = BiasedMatrixFactorizationALS(n_factors=4, n_iterations=3)

    student = StudentProfile(student_id="stu_test", name="Test Student")
    student.log_interaction("res_001", "completed", score=4.5)

    # Flag student as weak in algorithms (topic: algorithms)
    for i in range(4):
        student.record_quiz_answer("algorithms", f"q_algo_{i}", is_correct=False)

    cf.fit([student], CATALOG[:40])
    engine = HybridRecommendationEngine(bkt, cb, cf, CATALOG[:40], config)

    rec_for_you, strengthen_basics = engine.recommend(student, k=5)

    assert len(rec_for_you) == 5
    assert len(strengthen_basics) > 0  # Should flag items addressing algorithms weakness

    # Remedial items should carry explanations and valid SW score
    for item in strengthen_basics:
        assert item.sw_score > 0
        assert "algorithms" in item.reason.lower() or item.weak_topic_trigger == "algorithms"


def test_exposure_decay():
    config = RecommenderConfig(exposure_decay_delta=0.85)
    bkt = BayesianKnowledgeTracer(config)
    cb = ContentBasedRecommender(CATALOG[:20])
    cf = BiasedMatrixFactorizationALS(n_factors=4, n_iterations=3)

    student = StudentProfile(student_id="stu_decay", name="Decay Student")
    cf.fit([student], CATALOG[:20])
    engine = HybridRecommendationEngine(bkt, cb, cf, CATALOG[:20], config)

    # Initially score
    scores_initial = engine.get_candidate_scores(student, candidate_ids=["res_002"])
    score_0 = scores_initial["res_002"]["hybrid_score"]

    # Student ignores res_002 two times
    student.increment_exposure("res_002")
    student.increment_exposure("res_002")

    scores_decayed = engine.get_candidate_scores(student, candidate_ids=["res_002"])
    score_2 = scores_decayed["res_002"]["hybrid_score"]

    # Score should decay by (0.85)^2 = 0.7225
    assert score_2 < score_0
    assert pytest.approx(scores_decayed["res_002"]["decay"], 0.01) == 0.85 ** 2
