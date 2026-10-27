"""
State management module for the Streamlit web application.
Caches simulation cohort, models, and tracks active session student profile.
"""

import copy
from typing import Dict, List, Optional
import streamlit as st

from jaash_rs.config import RecommenderConfig, DEFAULT_CONFIG
from jaash_rs.domain.resources import CATALOG, LearningResource
from jaash_rs.models.bkt import BayesianKnowledgeTracer
from jaash_rs.models.collaborative import BiasedMatrixFactorizationALS
from jaash_rs.models.content_filtering import ContentBasedRecommender
from jaash_rs.models.hybrid_engine import HybridRecommendationEngine
from jaash_rs.profile.cohort_generator import SimulationCohort
from jaash_rs.profile.student_profile import StudentProfile


@st.cache_resource(show_spinner=False)
def get_cached_cohort_and_models(num_students: int = 100):
    """
    Initializes and caches the simulation cohort and pre-trained models.
    Uses 100 students for snappy interactive loading (400 available in full benchmark).
    """
    config = RecommenderConfig(sim_num_students=num_students)
    cohort = SimulationCohort(num_students=num_students, random_seed=42)

    bkt = BayesianKnowledgeTracer(config)
    cb = ContentBasedRecommender(cohort.resources)
    cf = BiasedMatrixFactorizationALS(
        n_factors=config.mf_latent_factors,
        reg_lambda=config.mf_regularization_lambda,
        n_iterations=15,
    )

    # Train ALS on cohort interactions
    cf.fit(cohort.students, cohort.resources)

    return cohort, bkt, cb, cf, config


def init_session_state():
    """Initializes Streamlit session state variables."""
    if "cohort" not in st.session_state:
        cohort, bkt, cb, cf, config = get_cached_cohort_and_models()
        st.session_state.cohort = cohort
        st.session_state.bkt = bkt
        st.session_state.cb = cb
        st.session_state.cf = cf
        st.session_state.config = copy.deepcopy(config)
        st.session_state.students_dict = {s.student_id: s for s in cohort.students}
        st.session_state.active_student_id = cohort.students[0].student_id

    if "quiz_active_topic" not in st.session_state:
        st.session_state.quiz_active_topic = "data_structures"

    if "quiz_step_feedback" not in st.session_state:
        st.session_state.quiz_step_feedback = None

    if "benchmarks_run" not in st.session_state:
        st.session_state.benchmarks_run = False
        st.session_state.table1_results = None
        st.session_state.table2_results = None
        st.session_state.sensitivity_results = None


def get_active_student() -> StudentProfile:
    init_session_state()
    return st.session_state.students_dict[st.session_state.active_student_id]


def get_engine(
    alpha: float, beta: float, gamma: float, theta: float, delta: float
) -> HybridRecommendationEngine:
    init_session_state()
    cfg = copy.deepcopy(st.session_state.config)
    cfg.alpha = alpha
    cfg.beta = beta
    cfg.gamma = gamma
    cfg.weakness_threshold_theta = theta
    cfg.exposure_decay_delta = delta

    return HybridRecommendationEngine(
        bkt_tracer=st.session_state.bkt,
        cb_recommender=st.session_state.cb,
        cf_recommender=st.session_state.cf,
        resources=st.session_state.cohort.resources,
        config=cfg,
    )
