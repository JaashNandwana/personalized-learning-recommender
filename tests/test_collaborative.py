"""
Unit tests for Collaborative Filtering Module.
"""

import numpy as np
import pytest

from jaash_rs.domain.resources import CATALOG
from jaash_rs.models.collaborative import BiasedMatrixFactorizationALS, PearsonNeighborhoodCF
from jaash_rs.profile.student_profile import StudentProfile


def test_pearson_cf():
    # 3 students with shared ratings
    s1 = StudentProfile(student_id="u1", name="User 1")
    s1.log_interaction("res_001", "rated", score=5.0)
    s1.log_interaction("res_002", "rated", score=4.0)

    s2 = StudentProfile(student_id="u2", name="User 2")
    s2.log_interaction("res_001", "rated", score=4.5)
    s2.log_interaction("res_002", "rated", score=4.0)
    s2.log_interaction("res_003", "rated", score=5.0)

    cf = PearsonNeighborhoodCF()
    cf.fit([s1, s2])

    pred = cf.predict_rating("u1", "res_003")
    assert 1.0 <= pred <= 5.0
    score = cf.score_resource("u1", "res_003")
    assert 0.0 <= score <= 1.0


def test_als_matrix_factorization():
    students = []
    # Create 10 synthetic students
    for i in range(10):
        s = StudentProfile(student_id=f"u_{i}", name=f"User {i}")
        for j in range(5):
            res_id = f"res_{j+1:03d}"
            s.log_interaction(res_id, "rated", score=float(3 + (i + j) % 3))
        students.append(s)

    als = BiasedMatrixFactorizationALS(n_factors=4, reg_lambda=1.0, n_iterations=5)
    als.fit(students, CATALOG[:10])

    pred = als.predict_rating("u_0", "res_001")
    assert 1.0 <= pred <= 5.0

    score = als.score_resource("u_0", "res_001")
    assert 0.0 <= score <= 1.0
