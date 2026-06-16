import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle

# Page config
st.set_page_config(
    page_title="Student Result Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 { font-size: 2.2rem; font-weight: 800; margin: 0; }
    .main-header p { font-size: 1rem; opacity: 0.85; margin-top: 0.5rem; }

    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border-left: 4px solid #2a5298;
        margin-bottom: 1rem;
    }
    .metric-card h3 { font-size: 0.85rem; color: #666; margin: 0 0 0.3rem 0; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-card p { font-size: 1.8rem; font-weight: 700; color: #1e3c72; margin: 0; }

    .result-pass {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.6rem;
        font-weight: 800;
    }
    .result-fail {
        background: linear-gradient(135deg, #cb2d3e, #ef473a);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.6rem;
        font-weight: 800;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1e3c72;
        border-bottom: 2px solid #2a5298;
        padding-bottom: 0.4rem;
        margin: 1.5rem 0 1rem 0;
    }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv('student_data.csv')

model_data = load_model()
model = model_data['model']
le_gender = model_data['le_gender']
le_parent = model_data['le_parent']
features = model_data['features']
df = load_data()

# Header
st.markdown("""
<div class="main-header">
    <h1>🎓 Student Result Predictor</h1>
    <p>ML-powered analytics to predict academic performance and identify at-risk students</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Predict Result", "📊 Data Analytics", "📈 Model Insights"])

# ─── TAB 1: PREDICTION ───────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-title">Enter Student Details</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        parental_education = st.selectbox("Parental Education", ["None", "High School", "Graduate", "Post-Graduate"])
        internet_access = st.selectbox("Internet Access at Home", ["Yes", "No"])
        extra_curricular = st.selectbox("Extra-Curricular Activities", ["Yes", "No"])

    with col2:
        study_hours = st.slider("Study Hours Per Day", 0.0, 12.0, 5.0, 0.5)
        attendance = st.slider("Attendance (%)", 0.0, 100.0, 75.0, 0.5)
        sleep_hours = st.slider("Sleep Hours Per Day", 3.0, 10.0, 7.0, 0.5)
        assignments = st.slider("Assignments Completed (out of 10)", 0, 10, 7)

    with col3:
        prev_score = st.slider("Previous Exam Score", 0.0, 100.0, 60.0, 0.5)
        math_score = st.slider("Math Score", 0.0, 100.0, 60.0, 0.5)
        science_score = st.slider("Science Score", 0.0, 100.0, 60.0, 0.5)
        english_score = st.slider("English Score", 0.0, 100.0, 65.0, 0.5)

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Result", use_container_width=True, type="primary")

    if predict_btn:
        gender_enc = le_gender.transform([gender])[0]
        parent_enc = le_parent.transform([parental_education])[0]
        internet_enc = 1 if internet_access == "Yes" else 0
        extra_enc = 1 if extra_curricular == "Yes" else 0

        input_data = np.array([[study_hours, attendance, prev_score, assignments,
                                 sleep_hours, math_score, science_score, english_score,
                                 extra_enc, internet_enc, gender_enc, parent_enc]])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        col_r1, col_r2, col_r3 = st.columns([1.5, 1, 1])

        with col_r1:
            if prediction == 1:
                st.markdown('<div class="result-pass">✅ PASS — Student is likely to pass!</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-fail">❌ FAIL — Student is at risk of failing!</div>', unsafe_allow_html=True)

        with col_r2:
            pass_prob = probability[1] * 100
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pass_prob,
                title={'text': "Pass Probability"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#2a5298"},
                    'steps': [
                        {'range': [0, 40], 'color': "#ffcccc"},
                        {'range': [40, 70], 'color': "#fff3cc"},
                        {'range': [70, 100], 'color': "#ccffcc"},
                    ],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 40}
                },
                number={'suffix': "%", 'font': {'size': 28}}
            ))
            fig.update_layout(height=220, margin=dict(t=30, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

        with col_r3:
            avg_score = (math_score + science_score + english_score) / 3
            st.markdown('<div class="section-title">Quick Summary</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><h3>Avg Subject Score</h3><p>{avg_score:.1f}/100</p></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><h3>Study Hours/Day</h3><p>{study_hours}h</p></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><h3>Attendance</h3><p>{attendance}%</p></div>', unsafe_allow_html=True)

        # Recommendations
        st.markdown('<div class="section-title">💡 Personalized Recommendations</div>', unsafe_allow_html=True)
        recs = []
        if attendance < 75: recs.append("📅 **Attendance is below 75%** — Aim for at least 85% attendance to significantly improve outcomes.")
        if study_hours < 4: recs.append("📚 **Study hours are low** — Try to study at least 4–5 hours daily.")
        if assignments < 7: recs.append("📝 **Assignment completion is low** — Complete at least 8 out of 10 assignments.")
        if sleep_hours < 6: recs.append("😴 **Sleep is insufficient** — Getting 7–8 hours of sleep improves focus and retention.")
        if avg_score < 50: recs.append("🧮 **Subject scores need improvement** — Focus on the weakest subject first.")
        if not recs: recs.append("🌟 **Great performance!** — Keep up the current habits and consistency.")
        for r in recs:
            st.markdown(f"- {r}")

# ─── TAB 2: ANALYTICS ────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><h3>Total Students</h3><p>{len(df)}</p></div>', unsafe_allow_html=True)
    with c2:
        pass_rate = df['result'].mean() * 100
        st.markdown(f'<div class="metric-card"><h3>Pass Rate</h3><p>{pass_rate:.1f}%</p></div>', unsafe_allow_html=True)
    with c3:
        avg_study = df['study_hours_per_day'].mean()
        st.markdown(f'<div class="metric-card"><h3>Avg Study Hours</h3><p>{avg_study:.1f}h</p></div>', unsafe_allow_html=True)
    with c4:
        avg_att = df['attendance_percentage'].mean()
        st.markdown(f'<div class="metric-card"><h3>Avg Attendance</h3><p>{avg_att:.1f}%</p></div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        grade_counts = df['grade'].value_counts().reset_index()
        grade_counts.columns = ['Grade', 'Count']
        grade_order = ['A', 'B', 'C', 'D', 'F']
        grade_counts['Grade'] = pd.Categorical(grade_counts['Grade'], categories=grade_order, ordered=True)
        grade_counts = grade_counts.sort_values('Grade')
        fig1 = px.bar(grade_counts, x='Grade', y='Count',
                      color='Grade',
                      color_discrete_map={'A':'#2ecc71','B':'#3498db','C':'#f39c12','D':'#e67e22','F':'#e74c3c'},
                      title="Grade Distribution")
        fig1.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        fig2 = px.scatter(df, x='study_hours_per_day', y='final_score',
                          color='grade',
                          color_discrete_map={'A':'#2ecc71','B':'#3498db','C':'#f39c12','D':'#e67e22','F':'#e74c3c'},
                          title="Study Hours vs Final Score",
                          labels={'study_hours_per_day': 'Study Hours/Day', 'final_score': 'Final Score'})
        fig2.update_layout(height=320)
        st.plotly_chart(fig2, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        att_bins = pd.cut(df['attendance_percentage'], bins=[0,50,60,70,80,90,100],
                          labels=['<50','50-60','60-70','70-80','80-90','90-100'])
        att_pass = df.groupby(att_bins, observed=True)['result'].mean().reset_index()
        att_pass.columns = ['Attendance Range', 'Pass Rate']
        att_pass['Pass Rate'] = (att_pass['Pass Rate'] * 100).round(1)
        fig3 = px.bar(att_pass, x='Attendance Range', y='Pass Rate',
                      title="Pass Rate by Attendance Range (%)",
                      color='Pass Rate',
                      color_continuous_scale='Blues')
        fig3.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        subj_avg = df[['math_score','science_score','english_score']].mean().reset_index()
        subj_avg.columns = ['Subject','Average Score']
        subj_avg['Subject'] = ['Math', 'Science', 'English']
        fig4 = px.pie(subj_avg, values='Average Score', names='Subject',
                      title="Average Score by Subject",
                      color_discrete_sequence=['#2a5298','#38a3a5','#57cc99'])
        fig4.update_layout(height=320)
        st.plotly_chart(fig4, use_container_width=True)

# ─── TAB 3: MODEL INSIGHTS ───────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">Model Performance & Feature Importance</div>', unsafe_allow_html=True)

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown('<div class="metric-card"><h3>Model Type</h3><p style="font-size:1.1rem">Random Forest</p></div>', unsafe_allow_html=True)
    with col_m2:
        st.markdown('<div class="metric-card"><h3>Accuracy</h3><p>87%</p></div>', unsafe_allow_html=True)
    with col_m3:
        st.markdown('<div class="metric-card"><h3>Training Data</h3><p>800 records</p></div>', unsafe_allow_html=True)

    feature_names_display = {
        'study_hours_per_day': 'Study Hours/Day',
        'attendance_percentage': 'Attendance %',
        'previous_score': 'Previous Score',
        'assignments_completed': 'Assignments Done',
        'sleep_hours': 'Sleep Hours',
        'math_score': 'Math Score',
        'science_score': 'Science Score',
        'english_score': 'English Score',
        'extra_curricular': 'Extra-Curricular',
        'internet_access': 'Internet Access',
        'gender_enc': 'Gender',
        'parent_enc': 'Parental Education'
    }

    importances = pd.Series(model.feature_importances_, index=features)
    importances.index = [feature_names_display.get(f, f) for f in features]
    importances = importances.sort_values()

    fig5 = px.bar(x=importances.values, y=importances.index,
                  orientation='h',
                  title="Feature Importance — What Affects Student Results Most?",
                  labels={'x': 'Importance Score', 'y': 'Feature'},
                  color=importances.values,
                  color_continuous_scale='Blues')
    fig5.update_layout(height=450, showlegend=False)
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("""
    <div style="background:#f0f4ff;border-radius:10px;padding:1.2rem 1.5rem;margin-top:1rem">
    <strong>📌 Key Insights from the Model:</strong><br><br>
    🔹 <b>Attendance</b> is the #1 predictor of student success — missing class is the biggest risk factor.<br>
    🔹 <b>Study hours per day</b> is the second most important factor.<br>
    🔹 <b>Previous academic performance</b> carries significant weight — early intervention matters.<br>
    🔹 <b>Assignments</b> completion is a strong behavioral indicator of engagement.<br>
    🔹 Gender and internet access have lower impact compared to behavioral factors.
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#888;font-size:0.85rem">
    Built with Python · Scikit-learn · Streamlit · Plotly &nbsp;|&nbsp; 
    <b>Vanga Manikanta Varaprasad</b> — AI/ML & Full Stack Developer
</div>
""", unsafe_allow_html=True)
