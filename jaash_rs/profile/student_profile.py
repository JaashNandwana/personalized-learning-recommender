"""
Student Profile module representing student state, learning history,
interactions, engagement matrix, and quiz answer records.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np


@dataclass
class QuizAttempt:
    topic_id: str
    question_id: str
    is_correct: bool
    time_spent_seconds: float = 30.0
    timestamp: float = 0.0


@dataclass
class ResourceInteraction:
    resource_id: str
    event_type: str  # 'opened', 'completed', 'bookmarked', 'abandoned', 'rated'
    implicit_score: float  # 1.0 to 5.0
    timestamp: float = 0.0


@dataclass
class StudentProfile:
    student_id: str
    name: str

    # Stated onboarding interests (topics & preferred formats)
    interested_topics: List[str] = field(default_factory=list)
    preferred_formats: List[str] = field(default_factory=lambda: ["video", "interactive"])
    preferred_difficulty: str = "intermediate"

    # Latent affinity vector (for simulation and collaborative modeling)
    latent_taste: Optional[np.ndarray] = None

    # Interactions: resource_id -> list of interactions
    interactions: Dict[str, List[ResourceInteraction]] = field(default_factory=dict)

    # Implicit engagement score: resource_id -> float [1.0, 5.0]
    engagement_scores: Dict[str, float] = field(default_factory=dict)

    # Quiz answer sequences per topic: topic_id -> list of QuizAttempt
    quiz_history: Dict[str, List[QuizAttempt]] = field(default_factory=dict)

    # Exposure count g(u, i): number of times resource i was recommended to student u without being opened
    exposure_counts: Dict[str, int] = field(default_factory=dict)

    # Derived preference vector over content feature dimensions
    preference_vector: Optional[np.ndarray] = None

    # Ground-truth hidden mastery per topic (used strictly in simulation benchmarks)
    true_hidden_mastery: Dict[str, bool] = field(default_factory=dict)

    def record_quiz_answer(self, topic_id: str, question_id: str, is_correct: bool, time_spent: float = 30.0) -> QuizAttempt:
        attempt = QuizAttempt(
            topic_id=topic_id,
            question_id=question_id,
            is_correct=is_correct,
            time_spent_seconds=time_spent
        )
        self.quiz_history.setdefault(topic_id, []).append(attempt)
        return attempt

    def log_interaction(self, resource_id: str, event_type: str, score: Optional[float] = None) -> ResourceInteraction:
        # Default implicit scores based on paper description:
        # opened: 1.0, partially completed/abandoned: 2.0, completed: 3.5, bookmarked: 4.5, rated/repeated: 5.0
        score_map = {
            "opened": 1.5,
            "abandoned": 2.0,
            "completed": 4.0,
            "bookmarked": 4.5,
            "rated": 5.0
        }
        computed_score = score if score is not None else score_map.get(event_type, 3.0)

        # Update best engagement score for this resource
        current_best = self.engagement_scores.get(resource_id, 0.0)
        self.engagement_scores[resource_id] = max(current_best, computed_score)

        # Reset exposure count once opened
        if event_type in ("opened", "completed", "bookmarked"):
            self.exposure_counts[resource_id] = 0

        inter = ResourceInteraction(
            resource_id=resource_id,
            event_type=event_type,
            implicit_score=computed_score
        )
        self.interactions.setdefault(resource_id, []).append(inter)
        return inter

    def increment_exposure(self, resource_id: str):
        """Increments g(u, i) when resource was shown to the student but not opened."""
        self.exposure_counts[resource_id] = self.exposure_counts.get(resource_id, 0) + 1

    def get_completed_resources(self) -> List[str]:
        return [res_id for res_id, score in self.engagement_scores.items() if score >= 3.0]

    def get_seen_resources(self) -> Set[str]:
        return set(self.engagement_scores.keys())
