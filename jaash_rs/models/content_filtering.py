"""
Content-Based Filtering Module.
Implements TF-IDF text representation and multi-attribute feature vectorization
for educational resources, along with student preference vector profiling and cosine similarity.
Follows Section IV-A of the paper.
"""

from typing import Dict, List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from jaash_rs.domain.curriculum import TOPIC_IDS, FORMATS, DIFFICULTIES
from jaash_rs.domain.resources import LearningResource, CATALOG
from jaash_rs.profile.student_profile import StudentProfile


class ContentBasedRecommender:
    def __init__(self, resources: List[LearningResource] = None, max_tfidf_features: int = 64):
        self.resources = resources or CATALOG
        self.resource_by_id: Dict[str, LearningResource] = {r.resource_id: r for r in self.resources}
        self.max_tfidf_features = max_tfidf_features

        # Precompute vocabulary and resource feature vectors
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_tfidf_features,
            stop_words="english",
            token_pattern=r"(?u)\b[a-zA-Z]{3,}\b"
        )
        self.resource_vectors: Dict[str, np.ndarray] = {}
        self._build_resource_vectors()

    def _build_resource_vectors(self):
        """
        Constructs normalized feature vector v_i for each resource:
        v_i = [ TF-IDF(title + description), Topics, Format, Difficulty ]
        """
        corpus = [f"{r.title}. {r.description}" for r in self.resources]
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus).toarray()

        topic_index = {t: idx for idx, t in enumerate(TOPIC_IDS)}
        format_index = {f: idx for idx, f in enumerate(FORMATS)}
        diff_index = {d: idx for idx, d in enumerate(DIFFICULTIES)}

        num_topics = len(TOPIC_IDS)
        num_formats = len(FORMATS)
        num_diffs = len(DIFFICULTIES)

        for i, res in enumerate(self.resources):
            # 1. TF-IDF features
            tfidf_feats = tfidf_matrix[i]

            # 2. Topic multi-hot features
            topic_feats = np.zeros(num_topics, dtype=float)
            for top in res.topics:
                if top in topic_index:
                    topic_feats[topic_index[top]] = 1.0

            # 3. Format one-hot features
            format_feats = np.zeros(num_formats, dtype=float)
            if res.format in format_index:
                format_feats[format_index[res.format]] = 1.0

            # 4. Difficulty one-hot features
            diff_feats = np.zeros(num_diffs, dtype=float)
            if res.difficulty in diff_index:
                diff_feats[diff_index[res.difficulty]] = 1.0

            # Concatenate and normalize
            raw_v = np.concatenate([tfidf_feats, topic_feats, format_feats, diff_feats])
            norm = np.linalg.norm(raw_v)
            v_i = raw_v / max(norm, 1e-9)

            self.resource_vectors[res.resource_id] = v_i

    def get_resource_vector(self, resource_id: str) -> Optional[np.ndarray]:
        return self.resource_vectors.get(resource_id)

    def compute_student_preference_vector(self, student: StudentProfile) -> np.ndarray:
        """
        Eq. (2) context: pu is the engagement-weighted mean of vectors of resources
        the student has completed (score >= 3.0).
        If cold-start (no completed items), builds preference from onboarding interests.
        """
        completed = student.get_completed_resources()
        feature_dim = next(iter(self.resource_vectors.values())).shape[0]

        if completed:
            weighted_sum = np.zeros(feature_dim, dtype=float)
            total_weight = 0.0

            for res_id in completed:
                if res_id in self.resource_vectors:
                    score = student.engagement_scores.get(res_id, 3.0)
                    weighted_sum += score * self.resource_vectors[res_id]
                    total_weight += score

            if total_weight > 0:
                p_u = weighted_sum / total_weight
                p_u_norm = np.linalg.norm(p_u)
                if p_u_norm > 1e-9:
                    student.preference_vector = p_u / p_u_norm
                    return student.preference_vector

        # Cold-start fallback: build pseudo-profile from onboarding stated interests
        topic_index = {t: idx for idx, t in enumerate(TOPIC_IDS)}
        format_index = {f: idx for idx, f in enumerate(FORMATS)}
        diff_index = {d: idx for idx, d in enumerate(DIFFICULTIES)}

        tfidf_dim = self.max_tfidf_features
        num_topics = len(TOPIC_IDS)
        num_formats = len(FORMATS)
        num_diffs = len(DIFFICULTIES)

        topic_feats = np.zeros(num_topics, dtype=float)
        for t in student.interested_topics:
            if t in topic_index:
                topic_feats[topic_index[t]] = 1.5

        format_feats = np.zeros(num_formats, dtype=float)
        for f in student.preferred_formats:
            if f in format_index:
                format_feats[format_index[f]] = 1.0

        diff_feats = np.zeros(num_diffs, dtype=float)
        if student.preferred_difficulty in diff_index:
            diff_feats[diff_index[student.preferred_difficulty]] = 1.0

        tfidf_dummy = np.zeros(tfidf_dim, dtype=float)
        p_u = np.concatenate([tfidf_dummy, topic_feats, format_feats, diff_feats])
        norm = np.linalg.norm(p_u)
        if norm > 1e-9:
            p_u = p_u / norm
        else:
            p_u = np.ones(feature_dim) / np.sqrt(feature_dim)

        student.preference_vector = p_u
        return p_u

    def score_resource(self, student: StudentProfile, resource_id: str) -> float:
        """
        Computes cosine similarity SCB(u, i) = (pu . vi) / (||pu|| * ||vi||)
        Rescaled strictly between 0.0 and 1.0.
        """
        v_i = self.resource_vectors.get(resource_id)
        if v_i is None:
            return 0.0

        p_u = student.preference_vector
        if p_u is None:
            p_u = self.compute_student_preference_vector(student)

        dot = float(np.dot(p_u, v_i))
        # Cosine similarity is in [-1, 1], non-negative feature space is typically in [0, 1]
        score = max(0.0, dot)
        return float(min(1.0, score))

    def score_all_candidates(self, student: StudentProfile, candidate_ids: List[str]) -> Dict[str, float]:
        """Scores multiple candidate resources."""
        p_u = self.compute_student_preference_vector(student)
        return {res_id: self.score_resource(student, res_id) for res_id in candidate_ids}
