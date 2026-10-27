"""
Bayesian Knowledge Tracing (BKT) Module.
Implements the two-state hidden Markov model for skill tracking described in Section IV-C.

Includes:
- Analytical one-step Bayesian posterior update and learning transition
- Topic-level parameter fitting via Grid Search Maximum Likelihood Estimation (MLE)
- Real-time mastery belief tracking P(Lt) and weakness scoring Wus = 1 - P(Lt)
- Running average baseline for comparison
- AUC, Precision, Recall, and F1 evaluation metrics
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

from jaash_rs.config import DEFAULT_CONFIG
from jaash_rs.domain.curriculum import TOPIC_IDS
from jaash_rs.profile.student_profile import StudentProfile, QuizAttempt


@dataclass
class BKTParameters:
    p_l0: float  # Initial mastery probability
    p_t: float   # Transition probability (learning rate)
    p_s: float   # Slip probability (mastered but answered wrongly)
    p_g: float   # Guess probability (unmastered but answered correctly)


class BayesianKnowledgeTracer:
    def __init__(self, config=DEFAULT_CONFIG):
        self.config = config
        # Parameters per topic
        self.topic_params: Dict[str, BKTParameters] = {}
        self._init_default_params()

    def _init_default_params(self):
        """Initializes standard baseline parameters for all curriculum topics."""
        for topic in TOPIC_IDS:
            self.topic_params[topic] = BKTParameters(
                p_l0=self.config.bkt_default_p_l0,
                p_t=self.config.bkt_default_p_t,
                p_s=self.config.bkt_default_p_s,
                p_g=self.config.bkt_default_p_g,
            )

    @staticmethod
    def compute_p_correct(p_l: float, p_s: float, p_g: float) -> float:
        """
        Calculates P(Ct): probability of a correct response given current mastery P(Lt).
        Eq. (6): P(Ct) = P(Lt)(1 - P(S)) + (1 - P(Lt))P(G)
        """
        p_c = p_l * (1.0 - p_s) + (1.0 - p_l) * p_g
        return float(np.clip(p_c, 1e-6, 1.0 - 1e-6))

    @staticmethod
    def update_posterior(p_l: float, is_correct: bool, p_s: float, p_g: float) -> float:
        """
        Revises mastery belief P(Lt | obs) given observed answer correctness.
        Eq. (7) for correct:   P(Lt|C) = [P(Lt)(1-P(S))] / [P(Lt)(1-P(S)) + (1-P(Lt))P(G)]
        Eq. (8) for incorrect: P(Lt|~C) = [P(Lt)P(S)] / [P(Lt)P(S) + (1-P(Lt))(1-P(G))]
        """
        if is_correct:
            numerator = p_l * (1.0 - p_s)
            denominator = numerator + (1.0 - p_l) * p_g
        else:
            numerator = p_l * p_s
            denominator = numerator + (1.0 - p_l) * (1.0 - p_g)

        if denominator < 1e-12:
            posterior = p_l
        else:
            posterior = numerator / denominator

        return float(np.clip(posterior, 1e-6, 1.0 - 1e-6))

    @staticmethod
    def apply_transition(p_l_obs: float, p_t: float) -> float:
        """
        Moves mastery belief forward by one learning opportunity.
        Eq. (9): P(Lt+1) = P(Lt | obs) + (1 - P(Lt | obs)) P(T)
        """
        next_p_l = p_l_obs + (1.0 - p_l_obs) * p_t
        return float(np.clip(next_p_l, 1e-6, 1.0 - 1e-6))

    def trace_sequence(
        self, topic_id: str, observations: List[bool]
    ) -> Tuple[List[float], List[float]]:
        """
        Traces a sequence of boolean answers for a single topic.
        Returns:
            p_lt_history: History of mastery estimates before each answer and final estimate
            p_ct_history: Predicted chance of answering correctly at each step
        """
        params = self.topic_params.get(
            topic_id,
            BKTParameters(
                self.config.bkt_default_p_l0,
                self.config.bkt_default_p_t,
                self.config.bkt_default_p_s,
                self.config.bkt_default_p_g,
            ),
        )

        curr_p_l = params.p_l0
        p_lt_history = [curr_p_l]
        p_ct_history = []

        for is_correct in observations:
            p_c = self.compute_p_correct(curr_p_l, params.p_s, params.p_g)
            p_ct_history.append(p_c)

            p_l_obs = self.update_posterior(curr_p_l, is_correct, params.p_s, params.p_g)
            curr_p_l = self.apply_transition(p_l_obs, params.p_t)
            p_lt_history.append(curr_p_l)

        return p_lt_history, p_ct_history

    def get_student_mastery(self, student: StudentProfile, topic_id: str) -> Tuple[float, int]:
        """
        Computes current P(Lt) for a student on a topic from their quiz history.
        Returns (current_mastery_probability, total_answers_count).
        """
        attempts = student.quiz_history.get(topic_id, [])
        if not attempts:
            params = self.topic_params.get(topic_id)
            p_l0 = params.p_l0 if params else self.config.bkt_default_p_l0
            return p_l0, 0

        obs = [a.is_correct for a in attempts]
        p_lt_history, _ = self.trace_sequence(topic_id, obs)
        return p_lt_history[-1], len(attempts)

    def get_weakness_profile(
        self, student: StudentProfile
    ) -> Dict[str, Dict[str, any]]:
        """
        Evaluates weakness across all topics for student u.
        Returns dict:
          topic_id -> {
             'mastery': float P(Lt)us,
             'weakness': float Wus = 1 - P(Lt)us,
             'is_weak': bool (P(Lt) < theta and count >= n_min),
             'is_unseen': bool (count < n_min),
             'count': int answers
          }
        """
        profile = {}
        theta = self.config.weakness_threshold_theta
        n_min = self.config.min_answers_n_min

        for topic in TOPIC_IDS:
            mastery, count = self.get_student_mastery(student, topic)
            weakness = 1.0 - mastery
            is_unseen = count < n_min
            is_weak = (mastery < theta) and (not is_unseen)

            profile[topic] = {
                "mastery": mastery,
                "weakness": weakness,
                "is_weak": is_weak,
                "is_unseen": is_unseen,
                "count": count,
            }
        return profile

    def fit_topic_mle(
        self,
        topic_id: str,
        student_sequences: List[List[bool]],
        grid_resolution: int = 5
    ) -> BKTParameters:
        """
        Fits BKT parameters (p_l0, p_t, p_s, p_g) for a topic using Grid Search MLE.
        Maximizes: LogLikelihood = sum_i sum_t [ y_it log P(C_it) + (1 - y_it) log(1 - P(C_it)) ]
        """
        l0_grid = np.linspace(0.20, 0.75, grid_resolution)
        t_grid = np.linspace(0.05, 0.25, grid_resolution)
        s_grid = np.linspace(0.05, 0.15, grid_resolution)
        g_grid = np.linspace(0.15, 0.30, grid_resolution)

        best_ll = -np.inf
        best_params = BKTParameters(0.5, 0.1, 0.1, 0.2)

        for l0 in l0_grid:
            for t in t_grid:
                for s in s_grid:
                    for g in g_grid:
                        ll = 0.0
                        for seq in student_sequences:
                            curr_pl = l0
                            for ans in seq:
                                pc = curr_pl * (1.0 - s) + (1.0 - curr_pl) * g
                                pc = np.clip(pc, 1e-9, 1.0 - 1e-9)
                                if ans:
                                    ll += np.log(pc)
                                    p_obs = (curr_pl * (1.0 - s)) / pc
                                else:
                                    ll += np.log(1.0 - pc)
                                    p_obs = (curr_pl * s) / (1.0 - pc)
                                curr_pl = p_obs + (1.0 - p_obs) * t

                        if ll > best_ll:
                            best_ll = ll
                            best_params = BKTParameters(float(l0), float(t), float(s), float(g))

        self.topic_params[topic_id] = best_params
        return best_params

    def evaluate_next_answer_auc(
        self, test_students: List[StudentProfile]
    ) -> Tuple[float, float]:
        """
        Evaluates Next Answer AUC for BKT and the Running Average baseline on test students.
        Returns: (bkt_auc, baseline_auc)
        """
        y_true = []
        bkt_preds = []
        base_preds = []

        for stu in test_students:
            for topic in TOPIC_IDS:
                attempts = stu.quiz_history.get(topic, [])
                if len(attempts) < 2:
                    continue

                obs = [a.is_correct for a in attempts]
                # BKT predictions
                _, p_ct_history = self.trace_sequence(topic, obs)

                # Baseline: running average of past answers (prior 0.5)
                running_correct = 0.0
                for step, (actual, bkt_prob) in enumerate(zip(obs, p_ct_history)):
                    if step == 0:
                        base_prob = 0.5
                    else:
                        base_prob = running_correct / step

                    y_true.append(1 if actual else 0)
                    bkt_preds.append(bkt_prob)
                    base_preds.append(base_prob)

                    if actual:
                        running_correct += 1.0

        if len(set(y_true)) < 2:
            return 0.5, 0.5

        bkt_auc = float(roc_auc_score(y_true, bkt_preds))
        base_auc = float(roc_auc_score(y_true, base_preds))
        return bkt_auc, base_auc

    def evaluate_weakness_detection(
        self, test_students: List[StudentProfile]
    ) -> Dict[str, float]:
        """
        Measures Weak Topic Detection Precision, Recall, and F1 score against
        the ground-truth unmastered topics.
        """
        y_true = []
        y_pred = []

        for stu in test_students:
            weak_profile = self.get_weakness_profile(stu)
            for topic in TOPIC_IDS:
                # True weak: student did NOT master the topic initially
                is_truly_weak = not stu.true_hidden_mastery.get(topic, False)
                pred_weak = weak_profile[topic]["is_weak"]

                y_true.append(1 if is_truly_weak else 0)
                y_pred.append(1 if pred_weak else 0)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
