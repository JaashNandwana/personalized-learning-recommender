"""
Configuration module for the Personalized Learning Content Recommendation System.
Parameters follow the specifications in:
"A Personalized Learning Content Recommendation System Using Hybrid Filtering and Knowledge Tracing"
(Jaash Nandwana, Ashmit Yadav, Anant Yadav)
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class RecommenderConfig:
    # Hybrid ranking weights (alpha + beta + gamma = 1.0)
    alpha: float = 0.4   # Collaborative Filtering weight
    beta: float = 0.3    # Content-Based Filtering weight
    gamma: float = 0.3   # Remediation (Knowledge Tracing) weight

    # Cold start weights (for new students without engagement history)
    cold_start_alpha: float = 0.0
    cold_start_beta: float = 0.7
    cold_start_gamma: float = 0.3

    # Exposure decay factor for unclicked items (delta in paper)
    exposure_decay_delta: float = 0.85

    # Weakness threshold theta: topics with P(Lt) < theta are flagged as weak
    weakness_threshold_theta: float = 0.60

    # Minimum answers required on a topic before weakness classification
    min_answers_n_min: int = 2

    # Collaborative Filtering Matrix Factorization (ALS) hyperparameters
    mf_latent_factors: int = 8
    mf_regularization_lambda: float = 2.0
    mf_iterations: int = 30

    # Top-K recommendations list size
    top_k: int = 5
    top_k_dashboard: int = 10

    # BKT Default Parameters (initial prior belief, learning transition, slip, guess)
    bkt_default_p_l0: float = 0.50
    bkt_default_p_t: float = 0.10
    bkt_default_p_s: float = 0.10
    bkt_default_p_g: float = 0.20

    # Simulation settings
    sim_num_students: int = 400
    sim_num_topics: int = 10
    sim_num_resources: int = 300
    sim_num_rounds: int = 4
    sim_learning_rate_from_resource: float = 0.25
    sim_remedial_click_bonus: float = 0.15


DEFAULT_CONFIG = RecommenderConfig()
