"""
Streamlit Web Application: Personalized Learning Content Recommendation System
Using Hybrid Filtering and Bayesian Knowledge Tracing.
Paper Authors: Jaash Nandwana, Ashmit Yadav, Anant Yadav (Manipal University Jaipur)
"""

import copy
import pandas as pd
import numpy as np
import streamlit as st

from jaash_rs.config import DEFAULT_CONFIG
from jaash_rs.domain.curriculum import TOPICS, TOPIC_DICT, TOPIC_IDS, FORMATS, DIFFICULTIES
from jaash_rs.domain.resources import CATALOG, CATALOG_BY_ID
from jaash_rs.domain.question_bank import QUESTIONS_BY_TOPIC, QUESTIONS_BY_ID, QuizQuestion
from jaash_rs.app.state import init_session_state, get_active_student, get_engine
from jaash_rs.simulation.experiment_runner import PaperExperimentRunner
from jaash_rs.profile.student_profile import StudentProfile

# Page Configuration
st.set_page_config(
    page_title="Personalized Learning Recommender (BKT + Hybrid)",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .paper-badge {
        background-color: #EEF2FF;
        color: #4338CA;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.5rem;
        border: 1px solid #C7D2FE;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .shelf-header {
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .badge-format {
        background-color: #E0E7FF;
        color: #3730A3;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-diff {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-weak {
        background-color: #FEE2E2;
        color: #B91C1C;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .remedial-box {
        background-color: #FEF2F2;
        border-left: 4px solid #EF4444;
        padding: 0.75rem 1rem;
        border-radius: 0 6px 6px 0;
        margin-bottom: 0.75rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
init_session_state()

# -------------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------------
st.sidebar.markdown("### 🎓 Student Profile")
active_student = get_active_student()

student_options = list(st.session_state.students_dict.keys())
selected_stu_id = st.sidebar.selectbox(
    "Select Student",
    options=student_options,
    index=student_options.index(st.session_state.active_student_id) if st.session_state.active_student_id in student_options else 0,
    format_func=lambda s_id: f"{st.session_state.students_dict[s_id].name} ({s_id})"
)
if selected_stu_id != st.session_state.active_student_id:
    st.session_state.active_student_id = selected_stu_id
    st.rerun()

student = get_active_student()
st.sidebar.markdown(f"**Preferred Formats:** {', '.join(student.preferred_formats)}")
st.sidebar.markdown(f"**Preferred Difficulty:** {student.preferred_difficulty.title()}")
st.sidebar.markdown(f"**Completed Items:** {len(student.get_completed_resources())}")

# Add Onboarding Form for New Custom Student
with st.sidebar.expander("➕ Onboard New Student Profile"):
    with st.form("onboard_form"):
        new_name = st.text_input("Student Name", value="Alex Rivera")
        new_topics = st.multiselect("Stated Interests", TOPIC_IDS, default=["python_basics", "data_structures"])
        new_fmts = st.multiselect("Preferred Formats", FORMATS, default=["video", "interactive"])
        new_diff = st.selectbox("Preferred Difficulty", DIFFICULTIES, index=1)
        submitted = st.form_submit_button("Create & Switch Profile")
        if submitted:
            new_id = f"stu_{len(st.session_state.students_dict)+1:03d}"
            new_profile = StudentProfile(
                student_id=new_id,
                name=new_name,
                interested_topics=new_topics,
                preferred_formats=new_fmts,
                preferred_difficulty=new_diff
            )
            st.session_state.students_dict[new_id] = new_profile
            st.session_state.active_student_id = new_id
            st.success(f"Created {new_name} ({new_id})")
            st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Recommender Hyperparameters")
st.sidebar.caption("Adjust institution scoring weights (sum to 1.0) and BKT parameters")

# Sliders for weights
alpha = st.sidebar.slider("α (Collaborative Filtering)", 0.0, 1.0, 0.40, 0.05)
beta = st.sidebar.slider("β (Content-Based Filtering)", 0.0, 1.0, 0.30, 0.05)
gamma = st.sidebar.slider("γ (Remediation / BKT)", 0.0, 1.0, 0.30, 0.05)

total_w = alpha + beta + gamma
if total_w > 0:
    alpha /= total_w
    beta /= total_w
    gamma /= total_w
st.sidebar.caption(f"Normalized Weights: **α={alpha:.2f}, β={beta:.2f}, γ={gamma:.2f}**")

theta = st.sidebar.slider("Weakness Threshold θ", 0.30, 0.90, 0.60, 0.05,
                          help="Topics with mastery P(Lt) < θ are flagged as weak.")
delta = st.sidebar.slider("Exposure Decay δ", 0.50, 1.00, 0.85, 0.05,
                          help="Decay multiplier for items repeatedly shown but ignored.")
top_k = st.sidebar.slider("Recommendations per Shelf (K)", 3, 10, 5, 1)

# Reset Interactions Button
if st.sidebar.button("🔄 Reset Current Student Interactions"):
    student.interactions.clear()
    student.engagement_scores.clear()
    student.quiz_history.clear()
    student.exposure_counts.clear()
    student.preference_vector = None
    st.sidebar.success("Reset completed!")
    st.rerun()

# -------------------------------------------------------------
# MAIN HEADER
# -------------------------------------------------------------
st.markdown('<div class="paper-badge">IEEE Conference Research Implementation</div>', unsafe_allow_html=True)
st.markdown('<div class="main-header">Personalized Learning Content Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Combining Collaborative Filtering, Content-Based Matching, and Bayesian Knowledge Tracing with Targeted Remediation</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Student Dashboard",
    "📝 Interactive Practice & Knowledge Tracing",
    "🔍 Scoring Engine Inspector",
    "📊 Paper Simulation & Benchmarks",
    "📚 Curriculum Catalog",
])

engine = get_engine(alpha, beta, gamma, theta, delta)

# =============================================================
# TAB 1: STUDENT DASHBOARD
# =============================================================
with tab1:
    col_dash1, col_dash2 = st.columns([1, 1])

    # Left: Student Profile & Weakness Radar/Bar Chart
    with col_dash1:
        st.subheader("📊 Estimated Knowledge State (BKT)")
        weak_profile = st.session_state.bkt.get_weakness_profile(student)

        # Build DataFrame for bar chart
        chart_data = []
        for t_id, t_info in weak_profile.items():
            t_name = TOPIC_DICT[t_id].name
            chart_data.append({
                "Topic": t_name,
                "Mastery P(Lt)": round(t_info["mastery"], 3),
                "Status": "Weak Topic" if t_info["is_weak"] else ("Unseen" if t_info["is_unseen"] else "Mastered"),
                "Answers": t_info["count"],
            })
        df_chart = pd.DataFrame(chart_data)

        # Display bar chart with threshold
        st.bar_chart(df_chart.set_index("Topic")["Mastery P(Lt)"], color="#4F46E5")
        st.caption(f"Horizontal reference line: Weakness threshold **θ = {theta:.2f}**. Topics below θ with ≥{DEFAULT_CONFIG.min_answers_n_min} answers are flagged for remediation.")

    # Right: Summary Badges
    with col_dash2:
        st.subheader("💡 Student Learning Status")
        weak_topics = [TOPIC_DICT[t].name for t, info in weak_profile.items() if info["is_weak"]]
        mastered_topics = [TOPIC_DICT[t].name for t, info in weak_profile.items() if not info["is_weak"] and not info["is_unseen"]]
        unseen_topics = [TOPIC_DICT[t].name for t, info in weak_profile.items() if info["is_unseen"]]

        c1, c2, c3 = st.columns(3)
        c1.metric("Weak Topics", len(weak_topics), delta=f"-{len(weak_topics)}" if weak_topics else "None", delta_color="inverse")
        c2.metric("Mastered", len(mastered_topics))
        c3.metric("Unseen / Untested", len(unseen_topics))

        if weak_topics:
            st.error(f"**Identified Weak Areas:** {', '.join(weak_topics)}")
        else:
            st.success("No critical weaknesses detected! All tested topics meet the mastery threshold.")

        st.info("💡 **Dual-Shelf Philosophy:** Rather than silently burying remedial material into standard recommendations (where students avoid them), the system explicitly separates them into **'Strengthen your basics'** with clear rationales.")

    st.markdown("---")

    # Generate Recommendations
    rec_for_you, strengthen_basics = engine.recommend(student, k=top_k, weights=(alpha, beta, gamma))

    # Shelf 1: "Recommended for you"
    st.markdown(f'<div class="shelf-header">🌟 Recommended For You (Top {top_k})</div>', unsafe_allow_html=True)
    st.caption("Ranked by overall hybrid score mixing your format preferences, peer engagement habits, and mastery needs.")

    if not rec_for_you:
        st.write("No recommendations available.")
    else:
        for idx, item in enumerate(rec_for_you, start=1):
            res = item.resource
            with st.container():
                st.markdown(f"#### {idx}. {res.title}")
                st.markdown(f"{res.description}")
                col_tags, col_btns = st.columns([3, 1])

                with col_tags:
                    tags_html = (
                        f'<span class="badge-format">Format: {res.format.title()}</span> '
                        f'<span class="badge-diff">Difficulty: {res.difficulty.title()}</span> '
                        f'<span>⏱️ {res.estimated_minutes} mins</span> | '
                        f'<span><b>Hybrid Score:</b> {item.hybrid_score:.3f} '
                        f'(SCF: {item.scf_score:.2f}, SCB: {item.scb_score:.2f}, SW: {item.sw_score:.2f}, Decay: {item.decay_factor:.2f})</span>'
                    )
                    st.markdown(tags_html, unsafe_allow_html=True)
                    if item.reason:
                        st.caption(f"💬 *{item.reason}*")

                with col_btns:
                    btn_open = st.button("📖 Open & Complete", key=f"open_rec_{res.resource_id}_{idx}")
                    if btn_open:
                        student.log_interaction(res.resource_id, "completed", score=4.5)
                        st.success(f"Completed '{res.title}'! Profile updated.")
                        st.rerun()

                st.markdown("<hr style='margin: 0.5rem 0;'>", unsafe_allow_html=True)

    # Shelf 2: "Strengthen your basics"
    st.markdown(f'<div class="shelf-header">🛠️ Strengthen Your Basics (Remediation Shelf)</div>', unsafe_allow_html=True)
    st.caption("Resources mapped directly to your detected weak topics with transparent pedagogical feedback.")

    if not strengthen_basics:
        st.success("🎉 You currently have no flagged weak topics! Excellent work.")
    else:
        for idx, item in enumerate(strengthen_basics, start=1):
            res = item.resource
            with st.container():
                st.markdown(f"#### {idx}. {res.title}")
                st.markdown(f'<div class="remedial-box">🎯 <b>Why recommended:</b> {item.reason}</div>', unsafe_allow_html=True)
                st.markdown(f"{res.description}")

                col_tags, col_btns = st.columns([3, 1])
                with col_tags:
                    tags_html = (
                        f'<span class="badge-format">Format: {res.format.title()}</span> '
                        f'<span class="badge-diff">Difficulty: {res.difficulty.title()}</span> '
                        f'<span class="badge-weak">Weakness Relevance SW: {item.sw_score:.2f}</span> | '
                        f'<span>Source: {res.source}</span>'
                    )
                    st.markdown(tags_html, unsafe_allow_html=True)

                with col_btns:
                    btn_rem_complete = st.button("✅ Complete Remedial Lab", key=f"open_rem_{res.resource_id}_{idx}")
                    if btn_rem_complete:
                        student.log_interaction(res.resource_id, "completed", score=5.0)
                        # Learning effect: slightly boost mastery for that weak topic
                        if item.weak_topic_trigger:
                            # Record 2 correct practice answers to reflect learning gain
                            student.record_quiz_answer(item.weak_topic_trigger, f"prac_{res.resource_id}_1", True)
                            student.record_quiz_answer(item.weak_topic_trigger, f"prac_{res.resource_id}_2", True)
                        st.success(f"Completed '{res.title}'! Knowledge tracing mastery improved.")
                        st.rerun()

                st.markdown("<hr style='margin: 0.5rem 0;'>", unsafe_allow_html=True)

# =============================================================
# TAB 2: INTERACTIVE PRACTICE QUIZ & REAL-TIME BKT
# =============================================================
with tab2:
    st.subheader("📝 Live Diagnostic Quiz & Real-Time Knowledge Tracing")
    st.write("Answer questions below to see Bayesian Knowledge Tracing revise its belief state **P(Lt)** after each response.")

    col_q1, col_q2 = st.columns([1, 1])

    with col_q1:
        chosen_topic = st.selectbox(
            "Select Topic to Practice",
            options=TOPIC_IDS,
            index=TOPIC_IDS.index(st.session_state.quiz_active_topic),
            format_func=lambda t: f"{TOPIC_DICT[t].name} ({t})"
        )
        st.session_state.quiz_active_topic = chosen_topic

        # Topic parameters
        bkt_params = st.session_state.bkt.topic_params.get(chosen_topic)
        current_mastery, num_ans = st.session_state.bkt.get_student_mastery(student, chosen_topic)

        st.markdown(f"""
        **Current Topic:** {TOPIC_DICT[chosen_topic].name}  
        **BKT Parameters:** $P(L_0)={bkt_params.p_l0:.2f}, P(T)={bkt_params.p_t:.2f}, P(S)={bkt_params.p_s:.2f}, P(G)={bkt_params.p_g:.2f}$  
        **Total Answers on Topic:** {num_ans}  
        **Current Estimated Mastery $P(L_t)$:** **{current_mastery:.3f}**  
        **Status:** {"🚨 WEAK (< 0.60)" if current_mastery < theta and num_ans >= 2 else ("⚪ UNTESTED" if num_ans < 2 else "✅ MASTERED")}
        """)

        # Fetch questions for this topic
        topic_questions = QUESTIONS_BY_TOPIC.get(chosen_topic, [])
        if topic_questions:
            # Pick next question based on answers count
            q_idx = num_ans % len(topic_questions)
            question = topic_questions[q_idx]

            st.markdown("---")
            st.markdown(f"**Question {num_ans + 1}:** {question.text}")
            selected_option_idx = st.radio(
                "Choose the best answer:",
                options=list(range(len(question.options))),
                format_func=lambda i: question.options[i],
                key=f"radio_q_{chosen_topic}_{question.question_id}"
            )

            btn_submit_ans = st.button("Submit Answer", type="primary")
            if btn_submit_ans:
                is_correct = (selected_option_idx == question.correct_index)

                # Prior
                prior_pl = current_mastery
                # Likelihood
                p_c = st.session_state.bkt.compute_p_correct(prior_pl, bkt_params.p_s, bkt_params.p_g)
                # Posterior
                posterior_pl = st.session_state.bkt.update_posterior(prior_pl, is_correct, bkt_params.p_s, bkt_params.p_g)
                # Transition
                next_pl = st.session_state.bkt.apply_transition(posterior_pl, bkt_params.p_t)

                # Record attempt
                student.record_quiz_answer(chosen_topic, question.question_id, is_correct)

                st.session_state.quiz_step_feedback = {
                    "is_correct": is_correct,
                    "explanation": question.explanation,
                    "prior": prior_pl,
                    "p_c": p_c,
                    "posterior": posterior_pl,
                    "next_pl": next_pl,
                    "question_text": question.text
                }
                st.rerun()

    with col_q2:
        st.subheader("🧮 Step-by-Step Bayesian Trace")
        fb = st.session_state.quiz_step_feedback
        if fb is None:
            st.info("Submit an answer on the left to inspect the Bayesian updating mathematics in real time.")
        else:
            if fb["is_correct"]:
                st.success("🎉 Correct Answer!")
            else:
                st.error("❌ Incorrect Answer.")

            st.markdown(f"**Explanation:** {fb['explanation']}")
            st.markdown("---")
            st.markdown("### Bayesian Knowledge Tracing Equations:")

            st.latex(r"P(C_t) = P(L_t)(1 - P(S)) + (1 - P(L_t))P(G)")
            st.write(f"Computed $P(C_t) = {fb['p_c']:.4f}$")

            if fb["is_correct"]:
                st.latex(r"P(L_t \mid C) = \frac{P(L_t)(1 - P(S))}{P(L_t)(1 - P(S)) + (1 - P(L_t))P(G)}")
                st.write(f"Posterior belief: $P(L_t \\mid C) = {fb['posterior']:.4f}$")
            else:
                st.latex(r"P(L_t \mid \bar{C}) = \frac{P(L_t) P(S)}{P(L_t) P(S) + (1 - P(L_t))(1 - P(G))}")
                st.write(f"Posterior belief: $P(L_t \\mid \\bar{{C}}) = {fb['posterior']:.4f}$")

            st.latex(r"P(L_{t+1}) = P(L_t \mid \text{obs}) + (1 - P(L_t \mid \text{obs})) P(T)")
            st.write(f"New state after learning transition $P(L_{{t+1}}) = $ **{fb['next_pl']:.4f}**")

            # Progress delta
            delta_val = fb["next_pl"] - fb["prior"]
            st.metric(
                label=f"Updated Mastery on {TOPIC_DICT[chosen_topic].name}",
                value=f"{fb['next_pl']:.3f}",
                delta=f"{delta_val:+.3f}"
            )

# =============================================================
# TAB 3: ENGINE SCORING INSPECTOR
# =============================================================
with tab3:
    st.subheader("🔍 Recommender Scoring Decomposition")
    st.markdown("""
    The hybrid scoring formula merges three normalized feature scores with exposure decay:
    """)
    st.latex(r"S(u, i) = \delta^{g(u, i)} \cdot \left[ \alpha S_{CF}(u, i) + \beta S_{CB}(u, i) + \gamma S_W(u, i) \right]")
    st.markdown(f"Current parameters: **α={alpha:.2f}, β={beta:.2f}, γ={gamma:.2f}, δ={delta:.2f}, θ={theta:.2f}**")

    # Get scored candidate details
    cand_scores = engine.get_candidate_scores(student, weights=(alpha, beta, gamma))
    rows = []
    for r_id, data in cand_scores.items():
        res = data["resource"]
        rows.append({
            "Resource ID": r_id,
            "Title": res.title,
            "Topics": ", ".join(res.topics),
            "Format": res.format,
            "Difficulty": res.difficulty,
            "S_CF (Collaborative)": round(data["scf"], 3),
            "S_CB (Content)": round(data["scb"], 3),
            "S_W (Remediation)": round(data["sw"], 3),
            "g(u,i) (Ignored)": data["g_ui"],
            "Decay Multiplier": round(data["decay"], 3),
            "Final Score S(u,i)": round(data["hybrid_score"], 3),
        })

    df_scores = pd.DataFrame(rows).sort_values(by="Final Score S(u,i)", ascending=False)
    st.dataframe(df_scores, use_container_width=True)

# =============================================================
# TAB 4: PAPER SIMULATION & BENCHMARKS
# =============================================================
with tab4:
    st.subheader("📊 Replication of Controlled Simulation Study (Section V)")
    st.markdown("""
    This section reproduces the experimental results from the research paper:
    - **Table I:** Knowledge Tracing and Weakness Detection Metrics
    - **Table II:** Recommendation Performance at K = 10
    - **Figure 4 & 5:** Dynamic mastery updates & remediation weight sensitivity curve
    """)

    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        run_bench = st.button("🚀 Run Simulation Benchmark (400 Students)", type="primary")

    if run_bench:
        with st.spinner("Running 400-student controlled simulation benchmark..."):
            runner = PaperExperimentRunner(cohort=st.session_state.cohort, config=st.session_state.config)
            st.session_state.table1_results = runner.run_table1_kt_evaluation()
            st.session_state.table2_results = runner.run_table2_benchmark(k=10)
            st.session_state.sensitivity_results = runner.run_sensitivity_sweep()
            st.session_state.benchmarks_run = True
            st.success("Simulation finished successfully!")

    if st.session_state.benchmarks_run:
        col_t1, col_t2 = st.columns(2)

        with col_t1:
            st.markdown("### 📋 Table I: Knowledge Tracing Results")
            t1 = st.session_state.table1_results
            df_t1 = pd.DataFrame([
                {"Measure": "Next answer AUC, BKT", "Value": f"{t1['bkt_auc']:.3f}"},
                {"Measure": "Next answer AUC, running average baseline", "Value": f"{t1['running_avg_baseline_auc']:.3f}"},
                {"Measure": "Weak topic detection precision", "Value": f"{t1['weak_detection_precision']:.3f}"},
                {"Measure": "Weak topic detection recall", "Value": f"{t1['weak_detection_recall']:.3f}"},
                {"Measure": "Weak topic detection F1", "Value": f"{t1['weak_detection_f1']:.3f}"},
            ])
            st.table(df_t1)

        with col_t2:
            st.markdown("### 📋 Table II: Recommendation Results at K = 10")
            t2 = st.session_state.table2_results
            df_t2 = pd.DataFrame(t2)
            st.table(df_t2.style.format({
                "P@10": "{:.3f}",
                "NDCG@10": "{:.3f}",
                "Cov@10": "{:.3f}",
                "Gain": lambda g: f"{g:.3f}" if g is not None else "n/a"
            }))

        st.markdown("---")
        st.markdown("### 📈 Figure 5: Sensitivity to Remediation Weight γ")
        sens_data = st.session_state.sensitivity_results
        df_sens = pd.DataFrame(sens_data)

        # Plot sensitivity
        st.line_chart(
            df_sens.set_index("gamma")[["coverage_at_10", "learning_gain", "precision_at_10"]]
        )
        st.caption("Coverage climbs rapidly with small γ (0.1 - 0.2), achieving over 90% weak topic coverage with minimal precision tradeoff.")
    else:
        st.info("Click 'Run Simulation Benchmark' above to execute the empirical study across 400 simulated students.")

# =============================================================
# TAB 5: CURRICULUM CATALOG
# =============================================================
with tab5:
    st.subheader("📚 Educational Resource Catalog (300 Items)")
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        f_topic = st.selectbox("Filter by Topic", ["All"] + TOPIC_IDS, format_func=lambda t: "All Topics" if t == "All" else TOPIC_DICT[t].name)
    with col_f2:
        f_format = st.selectbox("Filter by Format", ["All"] + FORMATS)
    with col_f3:
        f_diff = st.selectbox("Filter by Difficulty", ["All"] + DIFFICULTIES)

    filtered = CATALOG
    if f_topic != "All":
        filtered = [r for r in filtered if f_topic in r.topics]
    if f_format != "All":
        filtered = [r for r in filtered if r.format == f_format]
    if f_diff != "All":
        filtered = [r for r in filtered if r.difficulty == f_diff]

    st.write(f"Showing **{len(filtered)}** of {len(CATALOG)} resources")
    df_cat = pd.DataFrame([r.to_dict() for r in filtered])
    st.dataframe(df_cat, use_container_width=True)
