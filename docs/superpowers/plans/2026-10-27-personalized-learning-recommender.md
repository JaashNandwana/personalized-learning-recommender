# Personalized Learning Content Recommendation System Implementation Plan

**Feature / System:** Hybrid Filtering & Bayesian Knowledge Tracing Educational Recommender  
**Date:** 2026-10-27  
**Specification Reference:** IEEE Conference Paper by Jaash Nandwana, Ashmit Yadav, Anant Yadav — *A Personalized Learning Content Recommendation System Using Hybrid Filtering and Knowledge Tracing*

---

## 1. Overview & System Goals
The goal of this project is to implement an end-to-end Personalized Learning Content Recommendation System that eliminates the "one-size-fits-all" limitation of traditional learning platforms.

Standard collaborative or preference-only recommenders inadvertently trap struggling students by recommending only what they comfortably engage with (avoidance behavior). This system addresses this by tightly coupling:
1. **Bayesian Knowledge Tracing (BKT)** to dynamically estimate skill mastery $P(L_t)_{us}$ and identify weak topics ($W_{us} = 1 - P(L_t)_{us}$ when $P(L_t) < \theta$).
2. **Content-Based Filtering ($S_{CB}$)** using resource metadata, difficulty, format, and TF-IDF description vectors against student preference vectors $p_u$.
3. **Collaborative Filtering ($S_{CF}$)** using Biased Matrix Factorization (ALS) and Pearson Neighborhood correlation over implicit engagement scores $r_{ui} \in [1, 5]$.
4. **Hybrid Remediation Engine ($S(u, i)$)** that combines normalized scores with institutional weights $\alpha, \beta, \gamma$ and exposure decay $\delta^{g(u,i)}$.
5. **Dual-Shelf Dashboard**: Distinctly presenting **"Recommended for you"** and **"Strengthen your basics"** (with pedagogical explanations).
6. **Simulation & Evaluation Suite**: Reproducing the paper's 400-student controlled simulation (Table I, Table II, Fig. 4, and Fig. 5).
7. **Interactive Streamlit Web Application**: An interactive experience for students, instructors, and researchers.

---

## 2. Mathematical Architecture

### Content-Based Filtering
- Term weighting: $w_{t,d} = tf_{t,d} \cdot \log \frac{N}{df_t}$
- Resource feature vector $v_i = [\text{TF-IDF}(d_i), \text{Topic}(i), \text{Format}(i), \text{Difficulty}(i)]$
- Student profile $p_u = \frac{\sum_{i \in \text{Completed}_u} r_{ui} v_i}{\sum_{i \in \text{Completed}_u} r_{ui}}$ (or initialized from onboarding preferences for cold-start)
- Content Score:
  $$S_{CB}(u, i) = \frac{p_u \cdot v_i}{\|p_u\| \|v_i\|}$$

### Collaborative Filtering
- Implicit engagement score $r_{ui} \in [1, 5]$ derived from interactions (opened=1, partially completed=2, completed=3, bookmarked=4, revisits=5).
- Neighborhood model: Pearson correlation over co-rated items:
  $$sim(u, v) = \frac{\sum_{i \in I_{uv}} (r_{ui} - \bar{r}_u)(r_{vi} - \bar{r}_v)}{\sqrt{\sum_{i \in I_{uv}} (r_{ui} - \bar{r}_u)^2} \sqrt{\sum_{i \in I_{uv}} (r_{vi} - \bar{r}_v)^2}}$$
  $$\hat{r}_{ui} = \bar{r}_u + \frac{\sum_{v \in N_k(u)} sim(u, v)(r_{vi} - \bar{r}_v)}{\sum_{v \in N_k(u)} |sim(u, v)|}$$
- Biased Matrix Factorization (ALS):
  $$\min_{x, y, b} \sum_{(u, i) \in \mathcal{K}} (r_{ui} - \mu - b_u - b_i - y_i^T x_u)^2 + \lambda (\|x_u\|^2 + \|y_i\|^2 + b_u^2 + b_i^2)$$
  with latent factor dimension $d=8$, $\lambda=2.0$, and ALS convergence.
  Rescaled $\hat{r}_{ui} \to [0, 1]$ as $S_{CF}(u, i)$.

### Bayesian Knowledge Tracing (BKT)
- Parameters per topic $s$: Initial mastery $P(L_0)$, learning rate $P(T)$, slip $P(S)$, guess $P(G)$.
- Expected correctness:
  $$P(C_t) = P(L_t)(1 - P(S)) + (1 - P(L_t))P(G)$$
- Posterior update:
  $$P(L_t \mid C) = \frac{P(L_t)(1 - P(S))}{P(L_t)(1 - P(S)) + (1 - P(L_t))P(G)}$$
  $$P(L_t \mid \bar{C}) = \frac{P(L_t)P(S)}{P(L_t)P(S) + (1 - P(L_t))(1 - P(G))}$$
- Transition:
  $$P(L_{t+1}) = P(L_t \mid \text{obs}) + (1 - P(L_t \mid \text{obs}))P(T)$$
- Weakness metric:
  $$W_{us} = 1 - P(L_t)_{us}$$
  Topic $s$ is flagged as weak if $P(L_t)_{us} < \theta$ (default $\theta = 0.6$) and $n \ge n_{min}$ (default $n_{min} = 2$).

### Hybrid Recommendation with Remediation & Decay
- Remediation score:
  $$S_W(u, i) = \max_{s \in T(i)} W_{us} \cdot \mathbb{I}[P(L_t)_{us} < \theta]$$
- Per-student Min-Max normalization of $S_{CF}, S_{CB}, S_W$ to $[0, 1]$.
- Combined score with exposure decay:
  $$S(u, i) = \delta^{g(u, i)} [\alpha S_{CF}(u, i) + \beta S_{CB}(u, i) + \gamma S_W(u, i)]$$
  Default weights: $\alpha=0.4, \beta=0.3, \gamma=0.3$; decay parameter $\delta=0.85$; $g(u, i)$ counts unclicked impressions.
- Shelves:
  - Top-$K$ by $S(u, i) \to$ **"Recommended for you"**
  - Top-$K$ among $S_W(u, i) > 0 \to$ **"Strengthen your basics"** (with mastery stats & weakness rationale).

---

## 3. Proposed Directory & File Structure

```
jaash_rs/
├── __init__.py
├── config.py                 # System configuration, default weights (α, β, γ, θ, δ)
├── domain/
│   ├── __init__.py
│   ├── curriculum.py         # 10 Topics (Data Structures, Algorithms, Python, Linear Algebra, etc.)
│   ├── resources.py          # 300 Curated Educational Resources (Video, Reading, Interactive, Practice)
│   └── question_bank.py      # Rich quiz questions per topic with explanations & answer keys
├── models/
│   ├── __init__.py
│   ├── bkt.py                # Bayesian Knowledge Tracing with MLE fitting & real-time updates
│   ├── content_filtering.py  # TF-IDF & metadata feature vectorizer, cosine similarity
│   ├── collaborative.py      # Pearson correlation k-NN & Biased Matrix Factorization (ALS)
│   └── hybrid_engine.py      # Normalization, exposure decay, dual-shelf ranking
├── profile/
│   ├── __init__.py
│   ├── student_profile.py    # Preference vectors, engagement logs, response histories
│   └── cohort_generator.py   # Controlled 400-student simulation cohort generator (Section V-B)
├── simulation/
│   ├── __init__.py
│   ├── metrics.py            # P@K, NDCG@K, RMSE, MAE, Cov@K, AUC, F1, Learning Gain
│   └── experiment_runner.py  # Reproduction of Table I, Table II, Fig. 4, Fig. 5
├── app/
│   ├── __init__.py
│   ├── app.py                # Main Streamlit Dashboard with multi-tab interface
│   ├── components.py         # UI cards, gauges, KaTeX explainers, charts
│   └── state.py              # Session state manager for live interactions
├── tests/
│   ├── __init__.py
│   ├── test_bkt.py           # Unit tests for BKT updates, MLE fitting, AUC
│   ├── test_content.py       # Unit tests for TF-IDF and preference cosine matching
│   ├── test_collaborative.py # Unit tests for Pearson k-NN and ALS matrix factorization
│   ├── test_hybrid.py        # Unit tests for hybrid scoring, decay, dual-shelf output
│   └── test_e2e_simulation.py# End-to-end integration and simulation benchmark test
└── run_demo.py               # Fast CLI runner to execute evaluation or launch Streamlit
```

---

## 4. Step-by-Step Implementation Tasks

### Task 1: Curriculum, Resource Catalog, and Question Bank
- Define 10 high-impact Computer Science & Data Science topics:
  1. Python Fundamentals
  2. Data Structures
  3. Algorithms & Complexity
  4. Linear Algebra
  5. Probability & Statistics
  6. Machine Learning Foundations
  7. Deep Learning & Neural Networks
  8. Relational Databases & SQL
  9. Web Development & APIs
  10. System Design & Cloud Computing
- Generate 300 realistic educational resources with realistic titles, descriptions, format (`video`, `reading`, `interactive`, `practice`), difficulty (`beginner`, `intermediate`, `advanced`), and 1-2 topic tags.
- Provide a question bank with multiple questions per topic (questions, options, correct indices, explanations).

### Task 2: Data Models & Synthetic Cohort Generator
- Implement `StudentProfile`:
  - Stated interests, format preferences, difficulty preference.
  - History of resource interactions (opened, completed, bookmarked, abandoned, rating).
  - Per-topic quiz answer sequence `[(question_id, is_correct, time_spent), ...]`.
  - Repeated exposure tracking `g(u, i)`.
- Implement `CohortGenerator` matching Section V-B:
  - 400 students, latent affinity vectors (6 dimensions).
  - True hidden topic mastery drawn from $U(0.25, 0.75)$.
  - True BKT parameters drawn from uniform distributions described in paper.
  - Generates initial 40 interactions per student with avoidance bias (positive interest for mastered, negative for unmastered).
  - Train/test split (80/20).

### Task 3: Bayesian Knowledge Tracing (BKT) Engine
- Implement BKT math:
  - Analytical formula for $P(C_t)$, $P(L_t \mid C)$, $P(L_t \mid \bar{C})$, and $P(L_{t+1})$.
  - Grid search Maximum Likelihood Estimation (MLE) on training sequences to fit $(P(L_0), P(T), P(S), P(G))$ per topic.
  - Live one-step Bayesian update for real-time quiz answering.
  - Weakness calculation: $W_{us} = 1 - P(L_t)_{us}$; weak flag: $P(L_t)_{us} < \theta$ and $n \ge n_{min}$.
  - Baseline model: Running average of correctness.
  - Evaluation: Next-answer prediction AUC, weakness detection Precision, Recall, F1.

### Task 4: Content-Based Filtering Module
- Build TF-IDF vectorizer over title and description strings.
- One-hot/multi-hot encode format, difficulty, and topic tags into a unified feature vector $v_i$.
- Compute student preference vector $p_u$ (engagement-weighted mean of completed resources, or onboarding vector for cold-start).
- Cosine similarity calculation $S_{CB}(u, i)$.

### Task 5: Collaborative Filtering Module
- Implicit engagement matrix $R$ ($r_{ui} \in [1, 5]$).
- Implement Pearson correlation k-NN with significance/overlap weighting.
- Implement Biased Matrix Factorization via Alternating Least Squares (ALS) ($d=8, \lambda=2.0$, 30 rounds).
- Rescaling predictions to $[0, 1]$ as $S_{CF}(u, i)$.
- Prediction evaluation: RMSE and MAE on held-out test ratings.

### Task 6: Hybrid Recommendation Engine & Dual-Shelf Generator
- Compute candidate scores $S_{CF}, S_{CB}, S_W$.
- Min-Max normalize each scorer across unseen candidate resources for student $u$.
- Apply weights $\alpha, \beta, \gamma$ ($\alpha+\beta+\gamma=1$).
- Apply exposure decay: $S(u, i) = \delta^{g(u, i)} \cdot (\alpha S_{CF} + \beta S_{CB} + \gamma S_W)$.
- Generate dual shelves:
  - Top-$K$ items by $S(u, i)$ $\to$ "Recommended for you"
  - Top-$K$ items with $S_W > 0$ $\to$ "Strengthen your basics", accompanied by human-readable explanations (e.g., "Your mastery in Algorithms is 35% (below 60% target)").

### Task 7: Simulation Benchmark & Experiment Suite
- Execute the 4-round learning gain simulation from Section V-B:
  - 4 rounds, top-5 recommendation list per round.
  - Probability of opening item follows affinity + 0.15 remedial group bonus.
  - Learning transition: opening an unmastered item transitions topic to mastered with probability 0.25.
  - 2 quiz questions per topic update BKT between rounds.
- Benchmark 4 configurations (matching Table II):
  1. Popularity Baseline
  2. Collaborative Filtering only ($\alpha=1.0, \beta=0, \gamma=0$)
  3. CF + CB ($\alpha=0.57, \beta=0.43, \gamma=0$)
  4. Full Hybrid ($\alpha=0.4, \beta=0.3, \gamma=0.3$)
- Sensitivity analysis: vary $\gamma \in [0.0, 0.6]$ in steps of 0.1 (matching Fig. 5).
- Output publication-ready tables and verification reports.

### Task 8: Streamlit Interactive E2E Application
- Build a multi-tab web application:
  - **Tab 1: Student Dashboard**: Profile summary, mastery radar/bar chart, "Recommended for you" shelf, "Strengthen your basics" shelf with clear rationales, resource interaction buttons (Open, Complete, Bookmark, Rate) that trigger live updates.
  - **Tab 2: Interactive Practice Quiz**: Select a topic or take a diagnostic quiz; real-time BKT calculation showing prior, likelihood, and posterior after each question, with live threshold badge ("Weak" vs "Mastered").
  - **Tab 3: Engine Inspector & Parameter Tuning**: Live weight sliders ($\alpha, \beta, \gamma, \theta, \delta$), score breakdown table showing $(S_{CF}, S_{CB}, S_W, \text{Decay}, S)$ for all candidates.
  - **Tab 4: Paper Benchmark & Simulation**: Live execution and visual replication of Table I, Table II, and Fig. 4/5 curves.
  - **Tab 5: Resource Catalog Explorer**: Filterable directory of all 300 learning resources.

### Task 9: Unit Testing, Integration Testing & End-to-End Verification
- Write pytest suites for each module (`test_bkt.py`, `test_content.py`, `test_collaborative.py`, `test_hybrid.py`, `test_e2e_simulation.py`).
- Run all unit and integration tests to achieve 100% test pass rate.
- Launch and verify the Streamlit application.
