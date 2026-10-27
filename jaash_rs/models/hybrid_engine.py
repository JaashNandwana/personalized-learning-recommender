"""
Hybrid Recommendation Engine with Remediation and Exposure Decay.
Implements the core scoring pipeline of Section IV-D, Figure 3, and Algorithm 1.

Integrates:
- Collaborative Filtering (SCF)
- Content-Based Filtering (SCB)
- Remediation Scoring (SW) driven by Bayesian Knowledge Tracing
- Min-Max normalization per student across unseen candidates
- Exposure decay factor delta^g(u, i)
- Dual-shelf partition: "Recommended for you" and "Strengthen your basics" with explanations
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from jaash_rs.config import RecommenderConfig, DEFAULT_CONFIG
from jaash_rs.domain.curriculum import TOPIC_DICT, TOPIC_IDS
from jaash_rs.domain.resources import LearningResource, CATALOG, CATALOG_BY_ID
from jaash_rs.models.bkt import BayesianKnowledgeTracer
from jaash_rs.models.collaborative import BiasedMatrixFactorizationALS, PearsonNeighborhoodCF
from jaash_rs.models.content_filtering import ContentBasedRecommender
from jaash_rs.profile.student_profile import StudentProfile


@dataclass
class RecommendationItem:
    resource: LearningResource
    hybrid_score: float
    scf_score: float
    scb_score: float
    sw_score: float
    decay_factor: float
    rank: int
    weak_topic_trigger: Optional[str] = None
    reason: Optional[str] = None


class HybridRecommendationEngine:
    def __init__(
        self,
        bkt_tracer: BayesianKnowledgeTracer,
        cb_recommender: ContentBasedRecommender,
        cf_recommender: BiasedMatrixFactorizationALS,
        resources: List[LearningResource] = None,
        config: RecommenderConfig = DEFAULT_CONFIG,
    ):
        self.bkt = bkt_tracer
        self.cb = cb_recommender
        self.cf = cf_recommender
        self.resources = resources or CATALOG
        self.res_by_id: Dict[str, LearningResource] = {r.resource_id: r for r in self.resources}
        self.config = config

    def _compute_remediation_score(
        self, student: StudentProfile, resource: LearningResource, weak_profile: Dict[str, Dict]
    ) -> Tuple[float, Optional[str]]:
        """
        Eq. (12): SW(u, i) = max_{s in T(i)} [ Wus * I[P(Lt)us < theta] ]
        Returns (sw_score, triggering_weak_topic_id)
        """
        best_sw = 0.0
        trigger_topic = None

        for topic in resource.topics:
            t_info = weak_profile.get(topic)
            if t_info and t_info["is_weak"]:
                weakness = t_info["weakness"]
                if weakness > best_sw:
                    best_sw = weakness
                    trigger_topic = topic

        return float(best_sw), trigger_topic

    def _normalize_array(self, arr: np.ndarray) -> np.ndarray:
        """Min-max normalizes scores across candidate items per student."""
        if len(arr) == 0:
            return arr
        min_v = float(np.min(arr))
        max_v = float(np.max(arr))
        spread = max_v - min_v
        if spread < 1e-9:
            return np.ones_like(arr) * 0.5
        return (arr - min_v) / spread

    def get_candidate_scores(
        self,
        student: StudentProfile,
        candidate_ids: Optional[List[str]] = None,
        weights: Optional[Tuple[float, float, float]] = None,
    ) -> Dict[str, Dict]:
        """
        Computes all component scores (SCF, SCB, SW), normalizes them, applies weights and exposure decay.
        """
        # Determine candidate unseen resources
        seen = student.get_seen_resources()
        if candidate_ids is None:
            candidates = [r.resource_id for r in self.resources if r.resource_id not in seen]
            if not candidates:
                # If all seen, allow all
                candidates = [r.resource_id for r in self.resources]
        else:
            candidates = candidate_ids

        if not candidates:
            return {}

        # Knowledge tracing weakness profile
        weak_profile = self.bkt.get_weakness_profile(student)

        # Raw scores
        raw_scf = []
        raw_scb = []
        raw_sw = []
        triggers = {}

        # Detect cold start: if student has no completed items, switch to cold start weights
        completed_count = len(student.get_completed_resources())
        if weights is not None:
            alpha, beta, gamma = weights
        elif completed_count == 0:
            alpha = self.config.cold_start_alpha
            beta = self.config.cold_start_beta
            gamma = self.config.cold_start_gamma
        else:
            alpha = self.config.alpha
            beta = self.config.beta
            gamma = self.config.gamma

        for r_id in candidates:
            res = self.res_by_id[r_id]

            # 1. Collaborative filtering score
            scf_val = self.cf.score_resource(student.student_id, r_id)
            raw_scf.append(scf_val)

            # 2. Content-based filtering score
            scb_val = self.cb.score_resource(student, r_id)
            raw_scb.append(scb_val)

            # 3. Remediation score
            sw_val, trigger = self._compute_remediation_score(student, res, weak_profile)
            raw_sw.append(sw_val)
            triggers[r_id] = trigger

        # Normalization per student
        norm_scf = self._normalize_array(np.array(raw_scf))
        norm_scb = self._normalize_array(np.array(raw_scb))

        # For SW, if any item addresses a weakness, scale SW proportionally
        sw_arr = np.array(raw_sw)
        max_sw = float(np.max(sw_arr))
        if max_sw > 0:
            norm_sw = sw_arr / max_sw
        else:
            norm_sw = sw_arr

        # Exposure decay: delta^g(u, i)
        delta = self.config.exposure_decay_delta
        candidate_dict = {}

        for idx, r_id in enumerate(candidates):
            g_ui = student.exposure_counts.get(r_id, 0)
            decay = float(delta ** g_ui)

            scf = float(norm_scf[idx])
            scb = float(norm_scb[idx])
            sw = float(norm_sw[idx])

            # Eq. (13): S(u, i) = delta^g(u, i) * [alpha * SCF + beta * SCB + gamma * SW]
            final_score = decay * (alpha * scf + beta * scb + gamma * sw)

            candidate_dict[r_id] = {
                "hybrid_score": float(final_score),
                "scf": scf,
                "scb": scb,
                "sw": sw,
                "raw_sw": float(raw_sw[idx]),
                "decay": decay,
                "g_ui": g_ui,
                "weak_trigger": triggers[r_id],
                "resource": self.res_by_id[r_id],
            }

        return candidate_dict

    def recommend(
        self,
        student: StudentProfile,
        k: Optional[int] = None,
        weights: Optional[Tuple[float, float, float]] = None,
    ) -> Tuple[List[RecommendationItem], List[RecommendationItem]]:
        """
        Executes Algorithm 1:
        Returns:
            recommended_for_you: Top-K by S(u, i)
            strengthen_your_basics: Top-K with SW(u, i) > 0 with pedagogical rationales
        """
        k = k or self.config.top_k
        scored_candidates = self.get_candidate_scores(student, weights=weights)

        if not scored_candidates:
            return [], []

        # 1. "Recommended for you" shelf (Line 9: show top K by S)
        sorted_by_s = sorted(
            scored_candidates.items(),
            key=lambda x: x[1]["hybrid_score"],
            reverse=True,
        )

        rec_for_you: List[RecommendationItem] = []
        for rank, (r_id, data) in enumerate(sorted_by_s[:k], start=1):
            res = data["resource"]
            reason = f"Matches your interest in {res.format} resources and your learning pace."
            if data["sw"] > 0 and data["weak_trigger"]:
                t_name = TOPIC_DICT.get(data["weak_trigger"]).name
                reason += f" Also strengthens foundational gaps in {t_name}."

            rec_for_you.append(
                RecommendationItem(
                    resource=res,
                    hybrid_score=data["hybrid_score"],
                    scf_score=data["scf"],
                    scb_score=data["scb"],
                    sw_score=data["sw"],
                    decay_factor=data["decay"],
                    rank=rank,
                    weak_topic_trigger=data["weak_trigger"],
                    reason=reason,
                )
            )

        # 2. "Strengthen your basics" shelf (Line 10: show top K with SW > 0)
        remedial_items = [
            (r_id, data)
            for r_id, data in scored_candidates.items()
            if data["raw_sw"] > 0 and data["weak_trigger"] is not None
        ]

        # Rank remedial items primarily by weakness relevance SW and hybrid score
        remedial_items.sort(
            key=lambda x: (x[1]["raw_sw"] * 0.7 + x[1]["hybrid_score"] * 0.3),
            reverse=True,
        )

        strengthen_basics: List[RecommendationItem] = []
        weak_profile = self.bkt.get_weakness_profile(student)

        for rank, (r_id, data) in enumerate(remedial_items[:k], start=1):
            res = data["resource"]
            trig = data["weak_trigger"]
            t_obj = TOPIC_DICT.get(trig)
            t_name = t_obj.name if t_obj else trig
            t_info = weak_profile.get(trig, {})
            mastery_pct = int(round(t_info.get("mastery", 0.3) * 100))

            explanation = (
                f"You scored lower on recent {t_name} questions (estimated mastery: {mastery_pct}%, "
                f"below {int(self.config.weakness_threshold_theta * 100)}% target). "
                f"This {res.difficulty} {res.format} will solidify key concepts."
            )

            strengthen_basics.append(
                RecommendationItem(
                    resource=res,
                    hybrid_score=data["hybrid_score"],
                    scf_score=data["scf"],
                    scb_score=data["scb"],
                    sw_score=data["sw"],
                    decay_factor=data["decay"],
                    rank=rank,
                    weak_topic_trigger=trig,
                    reason=explanation,
                )
            )

        return rec_for_you, strengthen_basics

    def record_recommendation_impressions(
        self, student: StudentProfile, recommendations: List[RecommendationItem]
    ):
        """
        Logs that items were shown to student u.
        Increments exposure count g(u, i) so unclicked items decay over rounds.
        """
        for item in recommendations:
            student.increment_exposure(item.resource.resource_id)
