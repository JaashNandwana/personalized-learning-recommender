"""
Unit tests for Bayesian Knowledge Tracing (BKT) module.
"""

import pytest
from jaash_rs.models.bkt import BayesianKnowledgeTracer, BKTParameters
from jaash_rs.profile.student_profile import StudentProfile
from jaash_rs.domain.curriculum import TOPIC_IDS


def test_bkt_probability_bounds():
    tracer = BayesianKnowledgeTracer()
    p_c = tracer.compute_p_correct(p_l=0.5, p_s=0.1, p_g=0.2)
    assert 0.0 <= p_c <= 1.0

    post_corr = tracer.update_posterior(p_l=0.5, is_correct=True, p_s=0.1, p_g=0.2)
    assert post_corr > 0.5  # Correct answer should raise belief

    post_wrong = tracer.update_posterior(p_l=0.5, is_correct=False, p_s=0.1, p_g=0.2)
    assert post_wrong < 0.5  # Wrong answer should decrease belief

    trans = tracer.apply_transition(post_corr, p_t=0.1)
    assert trans >= post_corr  # Learning opportunity increases mastery


def test_bkt_sequence_tracing():
    tracer = BayesianKnowledgeTracer()
    topic = "python_basics"
    # A run of wrong answers
    obs_wrong = [False, False, False, False]
    p_l_hist_wrong, _ = tracer.trace_sequence(topic, obs_wrong)
    assert p_l_hist_wrong[-1] < 0.40  # Should be low mastery

    # A run of correct answers
    obs_correct = [True, True, True, True, True]
    p_l_hist_corr, _ = tracer.trace_sequence(topic, obs_correct)
    assert p_l_hist_corr[-1] > 0.70  # Should be high mastery


def test_student_weakness_profile():
    tracer = BayesianKnowledgeTracer()
    student = StudentProfile(student_id="test_01", name="Test Student")

    # Record 4 wrong answers for linear_algebra
    for i in range(4):
        student.record_quiz_answer("linear_algebra", f"q_{i}", is_correct=False)

    # Record 4 correct answers for python_basics
    for i in range(4):
        student.record_quiz_answer("python_basics", f"q_{i}", is_correct=True)

    # databases_sql has 0 answers (unseen)
    profile = tracer.get_weakness_profile(student)

    assert profile["linear_algebra"]["is_weak"] is True
    assert profile["linear_algebra"]["weakness"] > 0.5
    assert profile["python_basics"]["is_weak"] is False
    assert profile["databases_sql"]["is_unseen"] is True
    assert profile["databases_sql"]["is_weak"] is False  # Unseen topics are not flooded
