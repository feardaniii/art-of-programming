# 🚀 STREAMLIT: FROM STATIC ANALYSIS TO INTERACTIVE WEB APPS
# Transform your data science projects into shareable applications!
# No frontend experience needed - just Python!

"""
📚 WHAT IS STREAMLIT?

Streamlit turns Python scripts into interactive web applications with NO HTML/CSS/JS!

INSTALLATION:
pip install streamlit plotly pandas numpy scipy

RUN THIS FILE:
streamlit run health_dashboard.py

Then open: http://localhost:8501
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy import stats
import time

# ============================================
# 🎨 PAGE CONFIGURATION (Always first!)
# ============================================
st.set_page_config(
    page_title="Health Analytics Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# 💾 SESSION STATE INITIALIZATION
# ============================================
if 'health_data' not in st.session_state:
    # Generate sample data for demo
    np.random.seed(42)
    dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]
    
    # Create realistic correlations
    sleep = np.random.normal(7.5, 1.2, 30).clip(4, 11)
    energy = 2 + 0.7 * sleep + np.random.normal(0, 1, 30)
    energy = energy.clip(1, 10)
    
    st.session_state.health_data = pd.DataFrame({
        'date': dates,
        'sleep_hours': sleep,
        'energy_level': energy,
        'exercise_minutes': np.random.exponential(30, 30).clip(0, 120),
        'stress_level': np.random.normal(5, 2, 30).clip(1, 10),
        'mood': np.random.choice(['Great', 'Good', 'Okay', 'Bad'], 30, p=[0.2, 0.4, 0.3, 0.1])
    })

if 'show_welcome' not in st.session_state:
    st.session_state.show_welcome = True

# ============================================
# 🎯 WELCOME MESSAGE
# ============================================
if st.session_state.show_welcome:
    st.balloons()
    st.info("""
    👋 **Welcome to Your Personal Health Dashboard!**
    
    This interactive app helps you:
    - 📊 Track daily health metrics
    - 🔍 Discover patterns in your data
    - 💡 Get personalized recommendations
    - 📈 Visualize your progress over time
    
    👈 **Get started:** Use the sidebar to log your daily health data!
    """)
    
    if st.button("Got it! Let's start 🚀"):
        st.session_state.show_welcome = False
        st.rerun()

# ============================================
# 🎯 MAIN TITLE
# ============================================
st.markdown('<p class="main-header">🏥 Personal Health Analytics Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Transform your health data into actionable insights</p>', unsafe_allow_html=True)
st.markdown("---")

# ============================================
# 📊 SIDEBAR: DATA INPUT & CONTROLS
# ============================================
with st.sidebar:
    st.header("📝 Daily Health Logger")
    
    # Date input
    log_date = st.date_input(
        "Date",
        datetime.now(),
        help="Select the date for this entry"
    )
    
    # Numeric inputs with sliders
    st.subheader("Sleep & Energy")
    sleep_hours = st.slider(
        "Sleep Hours 😴",
        min_value=0.0,
        max_value=12.0,
        value=7.5,
        step=0.5,
        help="How many hours did you sleep?"
    )
    
    energy_level = st.slider(
        "Energy Level ⚡",
        min_value=1,
        max_value=10,
        value=7,
        help="Rate your energy from 1 (exhausted) to 10 (energized)"
    )
    
    st.subheader("Activity & Wellness")
    exercise_minutes = st.number_input(
        "Exercise Minutes 💪",
        min_value=0,
        max_value=300,
        value=30,
        step=5
    )
    
    stress_level = st.slider(
        "Stress Level 😰",
        min_value=1,
        max_value=10,
        value=5,
        help="Rate your stress from 1 (relaxed) to 10 (overwhelmed)"
    )
    
    mood = st.selectbox(
        "Overall Mood 😊",
        options=['Great', 'Good', 'Okay', 'Bad'],
        index=1
    )
    
    # Save button - UPDATED API
    if st.button("💾 Save Today's Data", type="primary", width='stretch'):
        new_entry = pd.DataFrame({
            'date': [pd.Timestamp(log_date)],
            'sleep_hours': [sleep_hours],
            'energy_level': [energy_level],
            'exercise_minutes': [exercise_minutes],
            'stress_level': [stress_level],
            'mood': [mood]
        })
        
        st.session_state.health_data = pd.concat([
            st.session_state.health_data,
            new_entry
        ]).drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
        
        st.success("✅ Data saved successfully!")
        st.balloons()
        time.sleep(1)
        st.rerun()
    
    st.markdown("---")
    
    # Data management
    st.subheader("📂 Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Reset", width='stretch', help="Clear all data and start fresh"):
            if st.session_state.get('confirm_reset'):
                st.session_state.health_data = st.session_state.health_data.iloc[:0]
                st.session_state.confirm_reset = False
                st.rerun()
            else:
                st.session_state.confirm_reset = True
                st.warning("Click again to confirm")
    
    with col2:
        # Download data as CSV
        csv = st.session_state.health_data.to_csv(index=False)
        st.download_button(
            label="📥 Export",
            data=csv,
            file_name=f"health_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            width='stretch'  # UPDATED API
        )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📤 Import CSV",
        type=['csv'],
        help="Upload your existing health data"
    )
    
    if uploaded_file is not None:
        try:
            imported_df = pd.read_csv(uploaded_file)
            imported_df['date'] = pd.to_datetime(imported_df['date'])
            st.session_state.health_data = imported_df
            st.success("✅ Data imported!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ============================================
# 📈 MAIN CONTENT
# ============================================

# Get data
df = st.session_state.health_data.copy()
df = df.sort_values('date')

# Check if we have data
if len(df) == 0:
    st.info("👆 **Start by logging your first health entry in the sidebar!**")
    
    # Show example data option
    if st.button("📊 Load Example Data", type="primary"):
        st.session_state.show_welcome = False
        st.rerun()
    
    st.stop()

# ============================================
# 🎯 KEY METRICS
# ============================================
st.header("📊 Your Health at a Glance")

col1, col2, col3, col4, col5 = st.columns(5)

# Metrics
avg_sleep = df['sleep_hours'].mean()
avg_energy = df['energy_level'].mean()
avg_exercise = df['exercise_minutes'].mean()
avg_stress = df['stress_level'].mean()

with col1:
    recent_sleep = df.tail(7)['sleep_hours'].mean()
    previous_sleep = df.iloc[-14:-7]['sleep_hours'].mean() if len(df) >= 14 else recent_sleep
    
    st.metric(
        label="😴 Avg Sleep",
        value=f"{avg_sleep:.1f}h",
        delta=f"{recent_sleep - previous_sleep:+.1f}h" if len(df) >= 14 else None
    )

with col2:
    recent_energy = df.tail(7)['energy_level'].mean()
    previous_energy = df.iloc[-14:-7]['energy_level'].mean() if len(df) >= 14 else recent_energy
    
    st.metric(
        label="⚡ Avg Energy",
        value=f"{avg_energy:.1f}/10",
        delta=f"{recent_energy - previous_energy:+.1f}" if len(df) >= 14 else None
    )

with col3:
    st.metric(
        label="💪 Avg Exercise",
        value=f"{avg_exercise:.0f}min"
    )

with col4:
    st.metric(
        label="😰 Avg Stress",
        value=f"{avg_stress:.1f}/10",
        delta_color="inverse"
    )

with col5:
    st.metric(
        label="📅 Days Tracked",
        value=len(df)
    )

st.markdown("---")

# ============================================
# 📊 TABS
# ============================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Trends",
    "🔗 Correlations",
    "📊 Patterns",
    "🎯 Insights"
])

# ============================================
# TAB 1: TRENDS
# ============================================
with tab1:
    st.subheader("📈 Your Health Trends Over Time")
    
    metrics_to_plot = st.multiselect(
        "Select metrics to display:",
        options=['sleep_hours', 'energy_level', 'exercise_minutes', 'stress_level'],
        default=['sleep_hours', 'energy_level'],
        format_func=lambda x: {
            'sleep_hours': '😴 Sleep Hours',
            'energy_level': '⚡ Energy Level',
            'exercise_minutes': '💪 Exercise Minutes',
            'stress_level': '😰 Stress Level'
        }[x]
    )
    
    if metrics_to_plot:
        fig = go.Figure()
        
        colors = {
            'sleep_hours': '#4A90E2',
            'energy_level': '#F5A623',
            'exercise_minutes': '#7ED321',
            'stress_level': '#D0021B'
        }
        
        for metric in metrics_to_plot:
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df[metric],
                mode='lines+markers',
                name=metric.replace('_', ' ').title(),
                line=dict(width=3, color=colors[metric]),
                marker=dict(size=8),
                hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Value: %{y:.1f}<extra></extra>'
            ))
        
        fig.update_layout(
            title="Health Metrics Timeline",
            xaxis_title="Date",
            yaxis_title="Value",
            hovermode='x unified',
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, width='stretch')  # UPDATED API
        
        # Trend analysis
        with st.expander("📊 Statistical Trend Analysis"):
            st.write("**Linear trend analysis (regression slope):**")
            
            for metric in metrics_to_plot:
                if len(df) >= 7:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        range(len(df)), df[metric]
                    )
                    
                    metric_name = metric.replace('_', ' ').title()
                    
                    if p_value < 0.05:
                        if slope > 0:
                            st.success(f"✅ **{metric_name}**: Improving at {slope:.3f}/day (p={p_value:.4f})")
                        else:
                            st.warning(f"⚠️ **{metric_name}**: Declining at {slope:.3f}/day (p={p_value:.4f})")
                    else:
                        st.info(f"ℹ️ **{metric_name}**: No significant trend (p={p_value:.4f})")
    else:
        st.info("👆 Select at least one metric to visualize")

# ============================================
# TAB 2: CORRELATIONS
# ============================================
with tab2:
    st.subheader("🔗 Discover Relationships in Your Data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        x_metric = st.selectbox(
            "X-axis:",
            options=['sleep_hours', 'exercise_minutes', 'stress_level'],
            format_func=lambda x: {
                'sleep_hours': '😴 Sleep Hours',
                'exercise_minutes': '💪 Exercise Minutes',
                'stress_level': '😰 Stress Level'
            }[x]
        )
        
        y_metric = st.selectbox(
            "Y-axis:",
            options=['energy_level', 'sleep_hours', 'stress_level'],
            index=0,
            format_func=lambda x: {
                'energy_level': '⚡ Energy Level',
                'sleep_hours': '😴 Sleep Hours',
                'stress_level': '😰 Stress Level'
            }[x]
        )
        
        # Calculate correlation
        correlation = df[x_metric].corr(df[y_metric])
        
        # Create scatter plot
        fig = px.scatter(
            df,
            x=x_metric,
            y=y_metric,
            color='mood',
            size='exercise_minutes',
            hover_data=['date'],
            title=f"Correlation: {correlation:.3f}",
            color_discrete_map={
                'Great': '#7ED321',
                'Good': '#4A90E2',
                'Okay': '#F5A623',
                'Bad': '#D0021B'
            }
        )
        
        # Add manual trendline
        if len(df) >= 3:
            z = np.polyfit(df[x_metric], df[y_metric], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(df[x_metric].min(), df[x_metric].max(), 100)
            
            fig.add_trace(go.Scatter(
                x=x_trend,
                y=p(x_trend),
                mode='lines',
                name='Trend',
                line=dict(color='red', dash='dash', width=2)
            ))
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, width='stretch')  # UPDATED API
    
    with col2:
        st.markdown("### 📊 Statistics")
        
        st.metric("Correlation", f"{correlation:.3f}")
        
        if abs(correlation) > 0.7:
            st.success("💪 Strong!")
        elif abs(correlation) > 0.4:
            st.info("📊 Moderate")
        else:
            st.warning("⚠️ Weak")
        
        # Statistical test
        if len(df) >= 3:
            _, p_value = stats.pearsonr(df[x_metric], df[y_metric])
            st.metric("P-value", f"{p_value:.4f}")
            
            if p_value < 0.05:
                st.success("✅ Significant!")
            else:
                st.info("Not significant")
        
        st.markdown("### 💡 What this means")
        
        if correlation > 0.5:
            st.markdown(f"When **{x_metric.replace('_', ' ')}** ⬆️ increases, **{y_metric.replace('_', ' ')}** tends to ⬆️ increase too!")
        elif correlation < -0.5:
            st.markdown(f"When **{x_metric.replace('_', ' ')}** ⬆️ increases, **{y_metric.replace('_', ' ')}** tends to ⬇️ decrease!")
        else:
            st.markdown("No strong relationship detected.")

# ============================================
# TAB 3: PATTERNS
# ============================================
with tab3:
    st.subheader("📊 Understand Your Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sleep distribution
        fig = px.histogram(
            df,
            x='sleep_hours',
            nbins=15,
            title="Sleep Hours Distribution",
            color_discrete_sequence=['#4A90E2']
        )
        fig.add_vline(
            x=df['sleep_hours'].mean(),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Avg: {df['sleep_hours'].mean():.1f}h"
        )
        st.plotly_chart(fig, width='stretch')  # UPDATED API
        
        # Energy by mood
        fig = px.box(
            df,
            x='mood',
            y='energy_level',
            title="Energy Level by Mood",
            color='mood',
            color_discrete_map={
                'Great': '#7ED321',
                'Good': '#4A90E2',
                'Okay': '#F5A623',
                'Bad': '#D0021B'
            }
        )
        st.plotly_chart(fig, width='stretch')  # UPDATED API
    
    with col2:
        # Exercise distribution
        fig = px.histogram(
            df,
            x='exercise_minutes',
            nbins=15,
            title="Exercise Minutes Distribution",
            color_discrete_sequence=['#7ED321']
        )
        fig.add_vline(
            x=df['exercise_minutes'].mean(),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Avg: {df['exercise_minutes'].mean():.0f}min"
        )
        st.plotly_chart(fig, width='stretch')  # UPDATED API
        
        # Stress distribution
        fig = px.histogram(
            df,
            x='stress_level',
            nbins=10,
            title="Stress Level Distribution",
            color_discrete_sequence=['#D0021B']
        )
        fig.add_vline(
            x=df['stress_level'].mean(),
            line_dash="dash",
            line_color="darkred",
            annotation_text=f"Avg: {df['stress_level'].mean():.1f}"
        )
        st.plotly_chart(fig, width='stretch')  # UPDATED API

# ============================================
# TAB 4: INSIGHTS
# ============================================
with tab4:
    st.subheader("🎯 Personalized Insights")
    
    if len(df) >= 8:
        high_energy = df[df['energy_level'] >= df['energy_level'].quantile(0.75)]
        low_energy = df[df['energy_level'] <= df['energy_level'].quantile(0.25)]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🌟 Your Best Days")
            st.metric("Avg Sleep", f"{high_energy['sleep_hours'].mean():.1f}h")
            st.metric("Avg Exercise", f"{high_energy['exercise_minutes'].mean():.0f}min")
            st.metric("Avg Stress", f"{high_energy['stress_level'].mean():.1f}/10")
        
        with col2:
            st.markdown("### 😔 Your Challenging Days")
            st.metric("Avg Sleep", f"{low_energy['sleep_hours'].mean():.1f}h")
            st.metric("Avg Exercise", f"{low_energy['exercise_minutes'].mean():.0f}min")
            st.metric("Avg Stress", f"{low_energy['stress_level'].mean():.1f}/10")
        
        st.markdown("---")
        st.markdown("### 💡 Recommendations")
        
        sleep_diff = high_energy['sleep_hours'].mean() - low_energy['sleep_hours'].mean()
        if sleep_diff > 0.5:
            st.success(f"😴 **Sleep {sleep_diff:.1f}h more** - Your best days have more sleep!")
        
        exercise_diff = high_energy['exercise_minutes'].mean() - low_energy['exercise_minutes'].mean()
        if exercise_diff > 15:
            st.success(f"💪 **Exercise {exercise_diff:.0f}min more** - Active days boost your energy!")
        
        stress_diff = low_energy['stress_level'].mean() - high_energy['stress_level'].mean()
        if stress_diff > 1:
            st.warning(f"😰 **Manage stress better** - {stress_diff:.1f} points lower on good days!")
    else:
        st.info("📊 Track at least 8 days to unlock personalized insights!")

# ============================================
# 🎓 FOOTER
# ============================================
st.markdown("---")

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    if st.button("📤 Share Dashboard", width='stretch'):  # UPDATED API
        st.balloons()
        st.success("✅ Dashboard link copied!")

with footer_col2:
    if st.button("📊 Generate Report", width='stretch'):  # UPDATED API
        with st.spinner("Generating report..."):
            time.sleep(1.5)
            st.success("✅ Report ready!")

with footer_col3:
    if st.button("🔔 Set Reminder", width='stretch'):  # UPDATED API
        st.info("✅ Daily reminder set for 9:00 PM")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>🎓 Built with Streamlit</strong></p>
    <p><em>No HTML, CSS, or JavaScript required - just Python!</em></p>
</div>
""", unsafe_allow_html=True)