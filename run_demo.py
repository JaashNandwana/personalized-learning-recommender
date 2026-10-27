"""
CLI entry point for the Personalized Learning Content Recommendation System.
Usage:
    python run_demo.py --app          # Launch Streamlit interactive web application
    python run_demo.py --benchmark    # Run full 400-student controlled simulation benchmark
    python run_demo.py --test         # Run full pytest test suite
"""

import argparse
import subprocess
import sys


def run_benchmark():
    from jaash_rs.config import RecommenderConfig
    from jaash_rs.profile.cohort_generator import SimulationCohort
    from jaash_rs.simulation.experiment_runner import PaperExperimentRunner

    print("\n==========================================================================")
    print("  PERSONALIZED LEARNING CONTENT RECOMMENDATION SYSTEM")
    print("  Replication of IEEE Conference Paper Simulation Study")
    print("  Authors: Jaash Nandwana, Ashmit Yadav, Anant Yadav (Manipal University Jaipur)")
    print("==========================================================================\n")

    print("[1/3] Generating controlled simulation cohort of 400 students...")
    config = RecommenderConfig(sim_num_students=400)
    cohort = SimulationCohort(num_students=400, random_seed=42)

    print("[2/3] Fitting Bayesian Knowledge Tracing & Collaborative Filtering ALS...")
    runner = PaperExperimentRunner(cohort=cohort, config=config)

    print("\n--------------------------------------------------------------------------")
    print("  TABLE I: KNOWLEDGE TRACING AND WEAKNESS DETECTION RESULTS")
    print("--------------------------------------------------------------------------")
    t1 = runner.run_table1_kt_evaluation()
    print(f"  Next answer AUC, BKT:                        {t1['bkt_auc']:.3f}")
    print(f"  Next answer AUC, running average baseline:   {t1['running_avg_baseline_auc']:.3f}")
    print(f"  Weak topic detection precision:              {t1['weak_detection_precision']:.3f}")
    print(f"  Weak topic detection recall:                 {t1['weak_detection_recall']:.3f}")
    print(f"  Weak topic detection F1:                     {t1['weak_detection_f1']:.3f}")
    print("--------------------------------------------------------------------------\n")

    print("[3/3] Running Table II benchmark comparisons at K = 10...")
    t2 = runner.run_table2_benchmark(k=10)
    print("--------------------------------------------------------------------------")
    print("  TABLE II: RECOMMENDATION RESULTS AT K = 10 (SIMULATION)")
    print("--------------------------------------------------------------------------")
    print(f"  {'Configuration':<26} {'P@10':<8} {'NDCG@10':<10} {'Cov@10':<9} {'Gain':<8}")
    print("  " + "-" * 62)
    for row in t2:
        gain_str = f"{row['Gain']:.3f}" if row['Gain'] is not None else "n/a"
        print(f"  {row['Configuration']:<26} {row['P@10']:<8.3f} {row['NDCG@10']:<10.3f} {row['Cov@10']:<9.3f} {gain_str:<8}")
    print("--------------------------------------------------------------------------\n")

    print("✔ Benchmark execution completed successfully!\n")


def launch_app():
    print("Launching Streamlit Web Application on http://localhost:8501 ...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "jaash_rs/app/app.py"])


def run_tests():
    print("Running pytest test suite...")
    subprocess.run([sys.executable, "-m", "pytest", "tests/"])


def main():
    parser = argparse.ArgumentParser(description="Personalized Learning Recommender CLI")
    parser.add_argument("--app", action="store_true", help="Launch Streamlit interactive web application")
    parser.add_argument("--benchmark", action="store_true", help="Run 400-student controlled simulation benchmark")
    parser.add_argument("--test", action="store_true", help="Run pytest test suite")

    args = parser.parse_args()

    if args.app:
        launch_app()
    elif args.benchmark:
        run_benchmark()
    elif args.test:
        run_tests()
    else:
        print("\nPersonalized Learning Content Recommendation System CLI")
        print("Usage:")
        print("  python run_demo.py --app          # Launch Streamlit web dashboard")
        print("  python run_demo.py --benchmark    # Run simulation benchmark (Table I & Table II)")
        print("  python run_demo.py --test         # Run all unit and integration tests\n")


if __name__ == "__main__":
    main()
