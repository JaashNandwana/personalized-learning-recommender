"""
Simulation Benchmark and Experiment Runner.
Replicates the empirical study and results reported in Section V of the paper:
- Table I: Knowledge Tracing and Weakness Detection Results (AUC, Precision, Recall, F1)
- Table II: Recommendation Results at K=10 (Popularity, CF only, CF+CB, Full Hybrid)
- Figure 5: Sensitivity analysis of the remediation weight gamma in [0.0, 0.6]
"""

import copy
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score

from jaash_rs.config import RecommenderConfig, DEFAULT_CONFIG
from jaash_rs.domain.curriculum import TOPIC_IDS
from jaash_rs.domain.resources import LearningResource, CATALOG, CATALOG_BY_ID
from jaash_rs.models.bkt import BayesianKnowledgeTracer
from jaash_rs.models.collaborative import BiasedMatrixFactorizationALS
from jaash_rs.models.content_filtering import ContentBasedRecommender
from jaash_rs.models.hybrid_engine import HybridRecommendationEngine
from jaash_rs.profile.cohort_generator import SimulationCohort
from jaash_rs.profile.student_profile import StudentProfile
from jaash_rs.simulation.metrics import precision_at_k, ndcg_at_k, coverage_at_k


class PaperExperimentRunner:
    def __init__(self, cohort: SimulationCohort = None, config: RecommenderConfig = DEFAULT_CONFIG):
        self.config = config
        self.cohort = cohort or SimulationCohort(num_students=config.sim_num_students, random_seed=42)
        self.resources = self.cohort.resources

        # Train/Test split: 70% train students, 30% test students (Section V-B)
        num_train = int(len(self.cohort.students) * 0.7)
        self.train_students = self.cohort.students[:num_train]
        self.test_students = self.cohort.students[num_train:]

        # For test students, split interaction sequence: first 80% train, last 20% held-out test
        self.prepared_test_students: List[StudentProfile] = []
        self.held_out_test_ratings: List[Dict[str, float]] = []

        for stu in self.test_students:
            stu_copy = copy.deepcopy(stu)
            items = list(stu.engagement_scores.items())
            num_tr = int(len(items) * 0.8)
            train_items = dict(items[:num_tr])
            held_out = dict(items[num_tr:])

            # Set engagement scores to train portion for evaluation
            stu_copy.engagement_scores = train_items
            self.prepared_test_students.append(stu_copy)
            self.held_out_test_ratings.append(held_out)

        # Shared models
        self.bkt = BayesianKnowledgeTracer(config)
        self.cb = ContentBasedRecommender(self.resources)
        self.cf = BiasedMatrixFactorizationALS(
            n_factors=config.mf_latent_factors,
            reg_lambda=config.mf_regularization_lambda,
            n_iterations=config.mf_iterations,
        )

        # Train BKT and CF models
        self._fit_models()

    def _fit_models(self):
        """Fits BKT on 70% train students, and CF ALS on training interactions."""
        # Fit BKT per topic
        for topic in TOPIC_IDS:
            seqs = [
                [a.is_correct for a in s.quiz_history.get(topic, [])]
                for s in self.train_students
                if len(s.quiz_history.get(topic, [])) > 0
            ]
            if seqs:
                self.bkt.fit_topic_mle(topic, seqs, grid_resolution=4)

        # Fit Collaborative Filtering ALS on train students + train portion of test students
        all_train_profiles = self.train_students + self.prepared_test_students
        self.cf.fit(all_train_profiles, self.resources)

    def run_table1_kt_evaluation(self) -> Dict[str, float]:
        """
        Replicates Table I:
        - Next answer AUC (BKT)
        - Next answer AUC (Running average baseline)
        - Weak topic detection Precision, Recall, F1
        """
        bkt_auc, base_auc = self.bkt.evaluate_next_answer_auc(self.test_students)
        detection = self.bkt.evaluate_weakness_detection(self.test_students)

        return {
            "bkt_auc": bkt_auc,
            "running_avg_baseline_auc": base_auc,
            "weak_detection_precision": detection["precision"],
            "weak_detection_recall": detection["recall"],
            "weak_detection_f1": detection["f1"],
        }

    def _simulate_learning_gain(
        self,
        students: List[StudentProfile],
        weights: Optional[Tuple[float, float, float]] = None,
        num_rounds: int = 4,
        top_k: int = 5,
        rng_seed: int = 42,
    ) -> float:
        """
        Simulates 4 rounds of recommendations to measure learning gain (Section V-B):
        - Chance of opening item follows affinity + 0.15 bonus for labelled remedial items
        - An opened item on an unmastered topic transitions topic to mastered with prob 0.25
        - 2 quiz questions per topic update BKT between rounds
        - Gain = share of initially unmastered topics that become mastered
        """
        sim_rng = np.random.default_rng(rng_seed)
        sim_students = copy.deepcopy(students)

        # Track initially weak pairs: (student_id, topic_id)
        initial_weak_pairs = set()
        for stu in sim_students:
            for top in TOPIC_IDS:
                if not stu.true_hidden_mastery.get(top, False):
                    initial_weak_pairs.add((stu.student_id, top))

        if not initial_weak_pairs:
            return 0.0

        engine = HybridRecommendationEngine(
            self.bkt, self.cb, self.cf, self.resources, self.config
        )

        for round_idx in range(num_rounds):
            for stu in sim_students:
                rec_for_you, remedial = engine.recommend(stu, k=top_k, weights=weights)

                # Remedial group presence depends on gamma > 0
                has_remediation = (weights is None) or (weights[2] > 0)
                remedial_ids = {r.resource.resource_id for r in remedial} if has_remediation else set()
                all_shown = rec_for_you

                for item in all_shown[:top_k]:
                    res = item.resource
                    # Calculate open chance from affinity
                    r_idx = int(res.resource_id.split("_")[1]) - 1
                    base_aff = float(np.dot(stu.latent_taste, self.cohort.resource_latents[r_idx]))

                    # Avoidance behavior: positive interest for mastered, negative for unmastered
                    topic_aff = sum(0.6 if stu.true_hidden_mastery.get(t, False) else -0.6 for t in res.topics)
                    bonus = self.config.sim_remedial_click_bonus if res.resource_id in remedial_ids else 0.0

                    logits = base_aff + topic_aff + bonus
                    open_prob = 1.0 / (1.0 + np.exp(-logits))

                    # Student opens item?
                    if sim_rng.random() < open_prob:
                        stu.log_interaction(res.resource_id, "completed", score=4.0)

                        # If resource covers an unmastered topic, chance to master it (prob 0.25)
                        for top in res.topics:
                            if not stu.true_hidden_mastery.get(top, False):
                                if sim_rng.random() < self.config.sim_learning_rate_from_resource:
                                    stu.true_hidden_mastery[top] = True

                # Post-round quiz: 2 questions per topic update BKT
                for top in TOPIC_IDS:
                    is_mastered = stu.true_hidden_mastery.get(top, False)
                    p_s = self.cohort.true_bkt_params[top]["p_s"]
                    p_g = self.cohort.true_bkt_params[top]["p_g"]

                    for q_i in range(2):
                        if is_mastered:
                            correct = bool(sim_rng.random() > p_s)
                        else:
                            correct = bool(sim_rng.random() < p_g)
                        stu.record_quiz_answer(top, f"sim_q_{round_idx}_{q_i}", correct)

        # Compute gain
        mastered_count = 0
        for stu_id, top in initial_weak_pairs:
            matched_stu = next(s for s in sim_students if s.student_id == stu_id)
            if matched_stu.true_hidden_mastery.get(top, False):
                mastered_count += 1

        gain = float(mastered_count / len(initial_weak_pairs))
        return gain

    def run_table2_benchmark(self, k: int = 10) -> List[Dict]:
        """
        Replicates Table II: Recommendation Results at K=10
        Evaluates 4 configurations:
        1. Popularity
        2. CF only (alpha=1.0, beta=0.0, gamma=0.0)
        3. CF + CB (alpha=0.57, beta=0.43, gamma=0.0)
        4. Full Hybrid (alpha=0.4, beta=0.3, gamma=0.3)
        """
        # Global popularity ranking
        item_counts = {}
        for s in self.train_students:
            for r_id in s.engagement_scores:
                item_counts[r_id] = item_counts.get(r_id, 0) + 1
        pop_sorted = [r_id for r_id, _ in sorted(item_counts.items(), key=lambda x: x[1], reverse=True)]

        configs = [
            ("Popularity", None),
            ("CF only", (1.0, 0.0, 0.0)),
            ("CF + CB (gamma = 0)", (0.57, 0.43, 0.0)),
            ("Full hybrid (gamma = 0.3)", (0.40, 0.30, 0.30)),
        ]

        results = []

        for name, weights in configs:
            precisions = []
            ndcgs = []
            coverages = []

            for stu, held_out in zip(self.prepared_test_students, self.held_out_test_ratings):
                rel_items = {r_id for r_id, score in held_out.items() if score >= 3.0}
                truly_weak = {t for t, m in stu.true_hidden_mastery.items() if not m}

                if name == "Popularity":
                    rec_ids = [r for r in pop_sorted if r not in stu.get_completed_resources()][:k]
                    rec_items = [CATALOG_BY_ID[r] for r in rec_ids if r in CATALOG_BY_ID]
                else:
                    engine = HybridRecommendationEngine(
                        self.bkt, self.cb, self.cf, self.resources, self.config
                    )
                    rec_for_you, _ = engine.recommend(stu, k=k, weights=weights)
                    rec_ids = [item.resource.resource_id for item in rec_for_you]
                    rec_items = [item.resource for item in rec_for_you]

                p_val = precision_at_k(rec_ids, rel_items, k=k)
                ndcg_val = ndcg_at_k(rec_ids, held_out, k=k)
                cov_val = coverage_at_k(rec_items, truly_weak, k=k)

                precisions.append(p_val)
                ndcgs.append(ndcg_val)
                coverages.append(cov_val)

            # Measure learning gain via simulation
            if name == "Popularity":
                gain_val = None
            else:
                gain_val = self._simulate_learning_gain(
                    self.test_students,
                    weights=weights,
                    num_rounds=4,
                    top_k=5,
                )

            results.append({
                "Configuration": name,
                "P@10": float(np.mean(precisions)),
                "NDCG@10": float(np.mean(ndcgs)),
                "Cov@10": float(np.mean(coverages)),
                "Gain": gain_val,
            })

        return results

    def run_sensitivity_sweep(self) -> List[Dict]:
        """
        Replicates Figure 5:
        Varies gamma in [0.0, 0.6] while keeping alpha and beta in ratio 4:3.
        """
        gamma_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        sweep_data = []

        for gamma in gamma_values:
            rem = 1.0 - gamma
            alpha = rem * (4.0 / 7.0)
            beta = rem * (3.0 / 7.0)
            weights = (alpha, beta, gamma)

            coverages = []
            precisions = []

            for stu, held_out in zip(self.prepared_test_students, self.held_out_test_ratings):
                rel_items = {r_id for r_id, score in held_out.items() if score >= 3.0}
                truly_weak = {t for t, m in stu.true_hidden_mastery.items() if not m}

                engine = HybridRecommendationEngine(self.bkt, self.cb, self.cf, self.resources, self.config)
                rec_for_you, _ = engine.recommend(stu, k=10, weights=weights)

                rec_ids = [item.resource.resource_id for item in rec_for_you]
                rec_items = [item.resource for item in rec_for_you]

                precisions.append(precision_at_k(rec_ids, rel_items, k=10))
                coverages.append(coverage_at_k(rec_items, truly_weak, k=10))

            gain = self._simulate_learning_gain(
                self.test_students,
                weights=weights,
                num_rounds=4,
                top_k=5,
            )

            sweep_data.append({
                "gamma": gamma,
                "coverage_at_10": float(np.mean(coverages)),
                "precision_at_10": float(np.mean(precisions)),
                "learning_gain": gain,
            })

        return sweep_data
