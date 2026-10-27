"""
End-to-end integration and simulation benchmark test.
"""

import pytest
from jaash_rs.config import RecommenderConfig
from jaash_rs.profile.cohort_generator import SimulationCohort
from jaash_rs.simulation.experiment_runner import PaperExperimentRunner


def test_e2e_simulation_pipeline():
    # Use smaller cohort for fast CI / test execution
    fast_config = RecommenderConfig(
        sim_num_students=30,
        mf_latent_factors=4,
        mf_iterations=5,
        mf_regularization_lambda=1.0,
    )
    cohort = SimulationCohort(num_students=30, random_seed=42)
    runner = PaperExperimentRunner(cohort=cohort, config=fast_config)

    # 1. Test Table I metrics
    table1 = runner.run_table1_kt_evaluation()
    assert "bkt_auc" in table1
    assert "weak_detection_f1" in table1
    assert 0.0 <= table1["bkt_auc"] <= 1.0
    assert 0.0 <= table1["weak_detection_f1"] <= 1.0

    # 2. Test Table II benchmark
    table2 = runner.run_table2_benchmark(k=5)
    assert len(table2) == 4
    for row in table2:
        assert "Configuration" in row
        assert "P@10" in row or "Cov@10" in row
        assert row["Cov@10"] >= 0.0

    # Verify that Full Hybrid achieves higher coverage than CF+CB
    cf_cb_row = next(r for r in table2 if "CF + CB" in r["Configuration"])
    hybrid_row = next(r for r in table2 if "Full hybrid" in r["Configuration"])
    assert hybrid_row["Cov@10"] >= cf_cb_row["Cov@10"]
