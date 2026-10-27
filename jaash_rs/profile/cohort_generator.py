"""
Cohort Generator for the 400-student simulation experiment described in Section V-B.
Synthesizes:
- True hidden mastery per student per topic
- Topic-specific ground truth BKT parameters
- Quiz responses simulating BKT dynamics
- Latent taste affinities with avoidance bias for unmastered topics
- Initial 40 resource interactions per student (80% train, 20% test)
"""

import random
from typing import Dict, List, Tuple
import numpy as np

from jaash_rs.domain.curriculum import TOPIC_IDS, FORMATS, DIFFICULTIES
from jaash_rs.domain.resources import LearningResource, CATALOG
from jaash_rs.profile.student_profile import StudentProfile, QuizAttempt


class SimulationCohort:
    def __init__(
        self,
        num_students: int = 400,
        resources: List[LearningResource] = None,
        random_seed: int = 42
    ):
        self.num_students = num_students
        self.resources = resources or CATALOG
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        self.py_rng = random.Random(random_seed)

        # True topic BKT parameters drawn from uniform ranges in Section V-B
        # P(L0) ~ U(0.25, 0.75), P(T) ~ U(0.05, 0.20), P(S) ~ U(0.05, 0.15), P(G) ~ U(0.15, 0.30)
        self.true_bkt_params: Dict[str, Dict[str, float]] = {}
        for topic in TOPIC_IDS:
            self.true_bkt_params[topic] = {
                "p_l0": float(self.rng.uniform(0.25, 0.75)),
                "p_t": float(self.rng.uniform(0.05, 0.20)),
                "p_s": float(self.rng.uniform(0.05, 0.15)),
                "p_g": float(self.rng.uniform(0.15, 0.30)),
            }

        # Resource latent representation (6 dimensions)
        self.resource_latents = self.rng.normal(0, 1.0, size=(len(self.resources), 6))
        # Normalize resource latents
        norms = np.linalg.norm(self.resource_latents, axis=1, keepdims=True)
        self.resource_latents = self.resource_latents / np.maximum(norms, 1e-9)

        self.students: List[StudentProfile] = []
        self._generate_cohort()

    def _generate_cohort(self):
        for idx in range(1, self.num_students + 1):
            student_id = f"stu_{idx:03d}"
            name = f"Student {idx:03d}"

            # Format preference
            preferred_fmt = self.py_rng.sample(FORMATS, k=2)
            preferred_diff = self.py_rng.choice(DIFFICULTIES)

            # 6-dimensional latent taste vector
            taste = self.rng.normal(0, 1.0, size=6)
            taste = taste / max(np.linalg.norm(taste), 1e-9)

            profile = StudentProfile(
                student_id=student_id,
                name=name,
                preferred_formats=preferred_fmt,
                preferred_difficulty=preferred_diff,
                latent_taste=taste
            )

            # 1. Determine True Hidden Mastery per topic
            for topic in TOPIC_IDS:
                p_l0 = self.true_bkt_params[topic]["p_l0"]
                is_mastered = bool(self.rng.random() < p_l0)
                profile.true_hidden_mastery[topic] = is_mastered

                # 2. Simulate between 4 and 11 quiz questions per topic
                num_questions = int(self.rng.integers(4, 12))  # 4 to 11 inclusive
                p_s = self.true_bkt_params[topic]["p_s"]
                p_g = self.true_bkt_params[topic]["p_g"]
                p_t = self.true_bkt_params[topic]["p_t"]

                current_mastery = is_mastered
                for q_i in range(num_questions):
                    q_id = f"q_{topic}_{q_i+1}"
                    # Chance of correct answer
                    if current_mastery:
                        # Mastery state: correct with 1 - p_s
                        is_correct = bool(self.rng.random() > p_s)
                    else:
                        # Non-mastered state: correct with guess p_g
                        is_correct = bool(self.rng.random() < p_g)

                    profile.record_quiz_answer(
                        topic_id=topic,
                        question_id=q_id,
                        is_correct=is_correct,
                        time_spent=float(self.rng.uniform(15.0, 75.0))
                    )

                    # State transition during testing (student might learn)
                    if not current_mastery and self.rng.random() < p_t:
                        current_mastery = True

            # 3. Simulate 40 Resource Interactions with avoidance behavior
            # "students show positive interest in topics they have mastered and negative
            # interest in topics they have not, which models avoidance behaviour"
            affinities = []
            for r_idx, res in enumerate(self.resources):
                # Base affinity from 6D latent dot product
                base_aff = float(np.dot(taste, self.resource_latents[r_idx]))

                # Avoidance / attraction factor from mastery of the resource's topics
                topic_bonus = 0.0
                for top in res.topics:
                    if profile.true_hidden_mastery.get(top, False):
                        topic_bonus += 0.8  # Attracted to mastered topics
                    else:
                        topic_bonus -= 0.8  # Avoids unmastered topics

                # Format alignment
                fmt_bonus = 0.3 if res.format in profile.preferred_formats else 0.0

                total_aff = base_aff + topic_bonus + fmt_bonus
                affinities.append(total_aff)

            affinities = np.array(affinities)
            # Softmax to form probability distribution over 300 resources
            # Scale logits with temperature
            exp_aff = np.exp((affinities - np.max(affinities)) / 0.8)
            probs = exp_aff / np.sum(exp_aff)

            # Sample 40 distinct resources without replacement
            sampled_indices = self.rng.choice(len(self.resources), size=40, replace=False, p=probs)

            # Assign implicit scores 1 to 5 based on affinity rank
            sampled_affs = affinities[sampled_indices]
            min_a, max_a = sampled_affs.min(), sampled_affs.max()
            span = max(max_a - min_a, 1e-5)

            for s_idx in sampled_indices:
                res = self.resources[s_idx]
                norm_aff = (affinities[s_idx] - min_a) / span
                # Map to 1 to 5 integer/half-integer score with small noise
                raw_score = 1.0 + 4.0 * norm_aff + self.rng.normal(0, 0.2)
                clamped_score = float(np.clip(np.round(raw_score * 2) / 2, 1.0, 5.0))

                event_type = "completed" if clamped_score >= 3.5 else ("opened" if clamped_score < 2.0 else "abandoned")
                profile.log_interaction(res.resource_id, event_type, score=clamped_score)

            self.students.append(profile)

    def get_train_test_split(self, train_ratio: float = 0.8) -> Tuple[List[StudentProfile], List[Dict[str, float]]]:
        """
        Splits interactions of each student into train (80%) and test (20%) sets.
        Returns students with train interactions and list of test interactions dict.
        """
        test_interactions = []
        for stu in self.students:
            items = list(stu.engagement_scores.items())
            num_train = int(len(items) * train_ratio)
            train_items = dict(items[:num_train])
            test_items = dict(items[num_train:])

            # Temporarily keep full in backup, or test dict
            test_interactions.append(test_items)

        return self.students, test_interactions
