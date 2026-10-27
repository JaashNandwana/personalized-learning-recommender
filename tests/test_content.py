"""
Unit tests for Content-Based Filtering Module.
"""

import numpy as np
import pytest

from jaash_rs.domain.resources import CATALOG
from jaash_rs.models.content_filtering import ContentBasedRecommender
from jaash_rs.profile.student_profile import StudentProfile


def test_resource_vector_generation():
    cb = ContentBasedRecommender(CATALOG[:50])
    v_1 = cb.get_resource_vector("res_001")
    assert v_1 is not None
    assert len(v_1) > 0
    # Vector should be unit normalized
    assert np.isclose(np.linalg.norm(v_1), 1.0, atol=1e-3)


def test_cold_start_profile():
    cb = ContentBasedRecommender(CATALOG[:50])
    student = StudentProfile(
        student_id="s_cold",
        name="Cold Student",
        interested_topics=["python_basics", "data_structures"],
        preferred_formats=["video"],
        preferred_difficulty="beginner",
    )
    p_u = cb.compute_student_preference_vector(student)
    assert p_u is not None
    assert np.isclose(np.linalg.norm(p_u), 1.0, atol=1e-3)

    # Score should be between 0 and 1
    score = cb.score_resource(student, "res_001")
    assert 0.0 <= score <= 1.0


def test_engagement_weighted_profile():
    cb = ContentBasedRecommender(CATALOG[:50])
    student = StudentProfile(student_id="s_active", name="Active Student")
    student.log_interaction("res_001", "completed", score=5.0)
    student.log_interaction("res_002", "completed", score=4.0)

    p_u = cb.compute_student_preference_vector(student)
    assert p_u is not None

    score_1 = cb.score_resource(student, "res_001")
    assert score_1 > 0.0
