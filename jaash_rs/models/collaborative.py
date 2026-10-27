"""
Collaborative Filtering Module.
Implements:
1. User-User Neighborhood Collaborative Filtering with Pearson Correlation (Eq. 3 & 4)
2. Biased Matrix Factorization via Alternating Least Squares (ALS) (Eq. 5)
3. Score rescaling to [0, 1] as SCF(u, i)
4. Evaluation measures: RMSE and MAE on held-out test ratings (Eq. 16)
Follows Section IV-B and Section V-D of the paper.
"""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from jaash_rs.config import DEFAULT_CONFIG
from jaash_rs.domain.resources import LearningResource, CATALOG
from jaash_rs.profile.student_profile import StudentProfile


class PearsonNeighborhoodCF:
    """
    User-User Collaborative Filtering using Pearson Correlation over common items.
    Suitable for small cohorts and direct interpretability.
    """
    def __init__(self, k_neighbors: int = 15):
        self.k_neighbors = k_neighbors
        self.student_ratings: Dict[str, Dict[str, float]] = {}
        self.user_means: Dict[str, float] = {}
        self.global_mean: float = 3.0

    def fit(self, students: List[StudentProfile]):
        self.student_ratings = {s.student_id: dict(s.engagement_scores) for s in students}
        all_ratings = []
        for s_id, ratings in self.student_ratings.items():
            if ratings:
                mean_val = float(np.mean(list(ratings.values())))
                self.user_means[s_id] = mean_val
                all_ratings.extend(ratings.values())
            else:
                self.user_means[s_id] = 3.0

        if all_ratings:
            self.global_mean = float(np.mean(all_ratings))

    def _pearson_similarity(self, u_id: str, v_id: str) -> float:
        """Eq. (3): Pearson correlation over common items I_uv."""
        u_ratings = self.student_ratings.get(u_id, {})
        v_ratings = self.student_ratings.get(v_id, {})

        common_items = set(u_ratings.keys()).intersection(v_ratings.keys())
        if len(common_items) < 2:
            return 0.0

        u_mean = self.user_means.get(u_id, self.global_mean)
        v_mean = self.user_means.get(v_id, self.global_mean)

        u_diff = np.array([u_ratings[i] - u_mean for i in common_items])
        v_diff = np.array([v_ratings[i] - v_mean for i in common_items])

        denom = np.sqrt(np.sum(u_diff ** 2)) * np.sqrt(np.sum(v_diff ** 2))
        if denom < 1e-9:
            return 0.0

        sim = float(np.sum(u_diff * v_diff) / denom)
        # Optional significance weighting if overlap is small
        if len(common_items) < 5:
            sim *= (len(common_items) / 5.0)
        return sim

    def predict_rating(self, student_id: str, resource_id: str) -> float:
        """Eq. (4): Predicts rating r_hat_ui."""
        u_mean = self.user_means.get(student_id, self.global_mean)
        u_ratings = self.student_ratings.get(student_id, {})

        if resource_id in u_ratings:
            return u_ratings[resource_id]

        similarities = []
        for other_id, other_ratings in self.student_ratings.items():
            if other_id == student_id or resource_id not in other_ratings:
                continue
            sim = self._pearson_similarity(student_id, other_id)
            if sim > 0:  # Only positive neighbors
                similarities.append((sim, other_ratings[resource_id], self.user_means.get(other_id, self.global_mean)))

        if not similarities:
            return u_mean

        # Select top-k neighbors
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_k = similarities[: self.k_neighbors]

        num = sum(sim * (r - v_mean) for sim, r, v_mean in top_k)
        denom = sum(abs(sim) for sim, _, _ in top_k)

        if denom < 1e-9:
            return u_mean

        pred = u_mean + (num / denom)
        return float(np.clip(pred, 1.0, 5.0))

    def score_resource(self, student_id: str, resource_id: str) -> float:
        """Rescales predicted rating to [0, 1] as SCF."""
        raw_pred = self.predict_rating(student_id, resource_id)
        return float(np.clip((raw_pred - 1.0) / 4.0, 0.0, 1.0))


class BiasedMatrixFactorizationALS:
    """
    Biased Matrix Factorization solved via Alternating Least Squares (ALS).
    Eq. (5):
    min sum (rui - mu - bu - bi - yi^T xu)^2 + lambda (||xu||^2 + ||yi||^2 + bu^2 + bi^2)
    """
    def __init__(
        self,
        n_factors: int = 8,
        reg_lambda: float = 2.0,
        n_iterations: int = 30,
        random_seed: int = 42
    ):
        self.n_factors = n_factors
        self.reg_lambda = reg_lambda
        self.n_iterations = n_iterations
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

        self.user_to_idx: Dict[str, int] = {}
        self.idx_to_user: Dict[int, str] = {}
        self.item_to_idx: Dict[str, int] = {}
        self.idx_to_item: Dict[int, str] = {}

        self.global_mean: float = 3.0
        self.user_biases: np.ndarray = np.array([])
        self.item_biases: np.ndarray = np.array([])
        self.user_factors: np.ndarray = np.array([[]])
        self.item_factors: np.ndarray = np.array([[]])

    def fit(self, students: List[StudentProfile], resources: List[LearningResource] = None):
        res_list = resources or CATALOG
        self.user_to_idx = {s.student_id: i for i, s in enumerate(students)}
        self.idx_to_user = {i: s.student_id for i, s in enumerate(students)}
        self.item_to_idx = {r.resource_id: j for j, r in enumerate(res_list)}
        self.idx_to_item = {j: r.resource_id for j, r in enumerate(res_list)}

        n_users = len(students)
        n_items = len(res_list)

        # Build observed entries
        user_items: Dict[int, List[Tuple[int, float]]] = {u: [] for u in range(n_users)}
        item_users: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(n_items)}
        all_ratings = []

        for s in students:
            u_idx = self.user_to_idx[s.student_id]
            for r_id, rating in s.engagement_scores.items():
                if r_id in self.item_to_idx:
                    i_idx = self.item_to_idx[r_id]
                    user_items[u_idx].append((i_idx, rating))
                    item_users[i_idx].append((u_idx, rating))
                    all_ratings.append(rating)

        self.global_mean = float(np.mean(all_ratings)) if all_ratings else 3.0

        # Initialize parameters
        self.user_biases = np.zeros(n_users, dtype=float)
        self.item_biases = np.zeros(n_items, dtype=float)
        self.user_factors = self.rng.normal(0, 0.1, size=(n_users, self.n_factors))
        self.item_factors = self.rng.normal(0, 0.1, size=(n_items, self.n_factors))

        eye_reg = self.reg_lambda * np.eye(self.n_factors)

        # Alternating Least Squares Optimization
        for iteration in range(self.n_iterations):
            # 1. Update User Biases and User Latent Factors
            for u in range(n_users):
                observed = user_items[u]
                if not observed:
                    continue

                item_indices = [idx for idx, _ in observed]
                ratings = np.array([r for _, r in observed])

                # Residual for bias update: r_ui - mu - b_i - y_i^T x_u
                y_sub = self.item_factors[item_indices]
                preds_wo_bu = self.global_mean + self.item_biases[item_indices] + np.dot(y_sub, self.user_factors[u])
                diff_b = ratings - preds_wo_bu
                self.user_biases[u] = np.sum(diff_b) / (len(observed) + self.reg_lambda)

                # Factor update: (Y^T Y + lambda I)^-1 Y^T (r_u - mu - b_u - b_items)
                resid_for_factors = ratings - (self.global_mean + self.user_biases[u] + self.item_biases[item_indices])
                A = np.dot(y_sub.T, y_sub) + eye_reg
                V = np.dot(y_sub.T, resid_for_factors)
                self.user_factors[u] = np.linalg.solve(A, V)

            # 2. Update Item Biases and Item Latent Factors
            for i in range(n_items):
                observed = item_users[i]
                if not observed:
                    continue

                user_indices = [idx for idx, _ in observed]
                ratings = np.array([r for _, r in observed])

                x_sub = self.user_factors[user_indices]
                preds_wo_bi = self.global_mean + self.user_biases[user_indices] + np.dot(x_sub, self.item_factors[i])
                diff_b = ratings - preds_wo_bi
                self.item_biases[i] = np.sum(diff_b) / (len(observed) + self.reg_lambda)

                resid_for_factors = ratings - (self.global_mean + self.user_biases[user_indices] + self.item_biases[i])
                A = np.dot(x_sub.T, x_sub) + eye_reg
                V = np.dot(x_sub.T, resid_for_factors)
                self.item_factors[i] = np.linalg.solve(A, V)

    def predict_rating(self, student_id: str, resource_id: str) -> float:
        """Eq. (5) prediction: r_hat_ui = mu + bu + bi + yi^T xu."""
        u_idx = self.user_to_idx.get(student_id)
        i_idx = self.item_to_idx.get(resource_id)

        if u_idx is None and i_idx is None:
            return self.global_mean
        if u_idx is None:
            return float(np.clip(self.global_mean + self.item_biases[i_idx], 1.0, 5.0))
        if i_idx is None:
            return float(np.clip(self.global_mean + self.user_biases[u_idx], 1.0, 5.0))

        pred = (
            self.global_mean
            + self.user_biases[u_idx]
            + self.item_biases[i_idx]
            + float(np.dot(self.item_factors[i_idx], self.user_factors[u_idx]))
        )
        return float(np.clip(pred, 1.0, 5.0))

    def score_resource(self, student_id: str, resource_id: str) -> float:
        """Rescales predicted rating to [0, 1] as SCF."""
        raw_pred = self.predict_rating(student_id, resource_id)
        return float(np.clip((raw_pred - 1.0) / 4.0, 0.0, 1.0))

    def evaluate_test_set(self, test_data: List[Tuple[str, str, float]]) -> Tuple[float, float]:
        """
        Computes RMSE (Eq. 16) and MAE over held-out test ratings.
        """
        if not test_data:
            return 0.0, 0.0

        errors = []
        for s_id, r_id, actual in test_data:
            pred = self.predict_rating(s_id, r_id)
            errors.append(pred - actual)

        errors = np.array(errors)
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mae = float(np.mean(np.abs(errors)))
        return rmse, mae
