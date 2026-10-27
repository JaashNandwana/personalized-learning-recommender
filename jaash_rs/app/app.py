"""
Streamlit Web Application: Personalized Learning Content Recommendation System
Using Hybrid Filtering and Bayesian Knowledge Tracing.
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
from jaash_rs.profile.student_profile import StudentProfile

# Page Configuration
st.set_page_config(
    page_title="Personalized Learning Recommender",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Theme-Adaptive Modern Styling (Dark & Light Mode Compatible)
st.markdown("""
<style>
    /* Typography */
    .main-title {
        font-size: 2.15rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin-bottom: 0.25rem;
        color: inherit;
    }
    .sub-title {
        font-size: 1.05rem;
        font-weight: 400;
        opacity: 0.85;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }
    .shelf-title {
        font-size: 1.3rem;
        font-weight: 700;
        margin-top: 1.25rem;
        margin-bottom: 0.35rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: inherit;
    }

    /* Badges */
    .badge-format {
        background: rgba(99, 102, 241, 0.16);
        color: #818CF8;
        border: 1px solid rgba(99, 102, 241, 0.35);
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-diff {
        background: rgba(245, 158, 11, 0.16);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.35);
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-weak {
        background: rgba(239, 68, 68, 0.16);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.35);
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-score {
        background: rgba(16, 185, 129, 0.16);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.35);
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-block;
    }

    /* Remedial Highlight Box */
    .remedial-box {
        background: rgba(239, 68, 68, 0.10);
        border-left: 4px solid #EF4444;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.75rem;
        font-size: 0.92rem;
        line-height: 1.5;
    }

    /* Score Meta Pill */
    .score-meta {
        font-size: 0.82rem;
        opacity: 0.85;
        font-family: monospace;
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
st.sidebar.markdown("### ⚙️ Engine Parameters")

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
st.markdown('<div class="main-title">Personalized Learning Content Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Adaptive learning platform combining Collaborative Filtering, Content-Based Matching, and Bayesian Knowledge Tracing with Targeted Remediation</div>', unsafe_allow_html=True)

# Tabs (Clean 4-tab interface, benchmark tab removed per user request)
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Student Dashboard",
    "📝 Interactive Practice & Knowledge Tracing",
    "🔍 Scoring Engine Inspector",
    "📚 Learning Resources",
])

engine = get_engine(alpha, beta, gamma, theta, delta)

# =============================================================
# TAB 1: STUDENT DASHBOARD
# =============================================================
with tab1:
    col_dash1, col_dash2 = st.columns([1, 1])

    # Left: Student Profile & Weakness Chart
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
        st.bar_chart(df_chart.set_index("Topic")["Mastery P(Lt)"], color="#6366F1")
        st.caption(f"Horizontal reference line: Weakness threshold **θ = {theta:.2f}**. Topics below θ with ≥{DEFAULT_CONFIG.min_answers_n_min} answers are flagged for remediation.")

    # Right: Summary Metrics
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

        st.info("💡 **Dual-Shelf Philosophy:** Instead of burying remedial material into standard recommendations (where students naturally avoid them), the system explicitly separates them into **'Strengthen your basics'** with clear rationales.")

    st.markdown("---")

    # Generate Recommendations
    rec_for_you, strengthen_basics = engine.recommend(student, k=top_k, weights=(alpha, beta, gamma))

    # Shelf 1: "Recommended for you"
    st.markdown(f'<div class="shelf-title">🌟 Recommended For You (Top {top_k})</div>', unsafe_allow_html=True)
    st.caption("Ranked by overall hybrid score matching your format preferences, peer engagement patterns, and mastery needs.")

    if not rec_for_you:
        st.write("No recommendations available.")
    else:
        for idx, item in enumerate(rec_for_you, start=1):
            res = item.resource
            with st.container(border=True):
                st.markdown(f"### {idx}. {res.title}")
                st.markdown(f"{res.description}")
                col_tags, col_btns = st.columns([3, 1])

                with col_tags:
                    tags_html = (
                        f'<span class="badge-format">Format: {res.format.title()}</span> '
                        f'<span class="badge-diff">Difficulty: {res.difficulty.title()}</span> '
                        f'<span class="badge-score">Hybrid Score: {item.hybrid_score:.3f}</span> | '
                        f'<span>⏱️ {res.estimated_minutes} mins</span> '
                        f'<span class="score-meta">(SCF: {item.scf_score:.2f}, SCB: {item.scb_score:.2f}, SW: {item.sw_score:.2f}, Decay: {item.decay_factor:.2f})</span>'
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

    # Shelf 2: "Strengthen your basics"
    st.markdown(f'<div class="shelf-title">🛠️ Strengthen Your Basics (Remediation Shelf)</div>', unsafe_allow_html=True)
    st.caption("Resources mapped directly to your detected weak topics with transparent pedagogical explanations.")

    if not strengthen_basics:
        st.success("🎉 You currently have no flagged weak topics! Excellent work.")
    else:
        for idx, item in enumerate(strengthen_basics, start=1):
            res = item.resource
            with st.container(border=True):
                st.markdown(f"### {idx}. {res.title}")
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
    st.dataframe(df_scores, width='stretch')

# =============================================================
# TAB 4: LEARNING RESOURCES CATALOG
# =============================================================
with tab4:
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
    st.dataframe(df_cat, width='stretch')
