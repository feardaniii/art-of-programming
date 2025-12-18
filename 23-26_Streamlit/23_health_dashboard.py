import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Page configuration
st.set_page_config(
    page_title="🏥 Personal Health Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for data persistence
if 'health_data' not in st.session_state:
    st.session_state.health_data = pd.DataFrame(columns=[
        'date', 'sleep_hours', 'sleep_quality', 'exercise_minutes', 
        'exercise_type', 'mood_score', 'energy_level', 'water_glasses'
    ])

# Custom CSS for better styling
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 0.5rem;
    color: white;
    text-align: center;
}
.insight-box {
    background: #f0f9ff;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #3b82f6;
}
</style>
""", unsafe_allow_html=True)

# Header
st.title("🏥 Personal Health Dashboard")
st.markdown("*Transform scattered health data into actionable insights*")

# Sidebar for data input
with st.sidebar:
    st.header("📝 Daily Health Log")
    
    # Date input
    log_date = st.date_input("Date", datetime.now().date())
    
    # Sleep tracking
    st.subheader("😴 Sleep")
    sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.5, 0.5)
    sleep_quality = st.slider("Sleep Quality (1-10)", 1, 10, 7)
    
    # Exercise tracking
    st.subheader("🏃‍♀️ Exercise")
    exercise_type = st.selectbox("Exercise Type", 
        ["None", "Running", "Cycling", "Swimming", "Gym", "Yoga", "Walking", "Other"])
    exercise_minutes = st.slider("Exercise Duration (minutes)", 0, 180, 30)
    
    # Wellness tracking
    st.subheader("😊 Wellness")
    mood_score = st.slider("Mood Score (1-10)", 1, 10, 7)
    energy_level = st.slider("Energy Level (1-10)", 1, 10, 7)
    water_glasses = st.slider("Water Glasses", 0, 15, 8)
    
    # Save data button
    if st.button("💾 Save Today's Data", type="primary"):
        new_data = {
            'date': log_date,
            'sleep_hours': sleep_hours,
            'sleep_quality': sleep_quality,
            'exercise_minutes': exercise_minutes,
            'exercise_type': exercise_type,
            'mood_score': mood_score,
            'energy_level': energy_level,
            'water_glasses': water_glasses
        }
        
        # Remove existing data for this date and add new
        st.session_state.health_data = st.session_state.health_data[
            st.session_state.health_data['date'] != log_date
        ]
        st.session_state.health_data = pd.concat([
            st.session_state.health_data,
            pd.DataFrame([new_data])
        ], ignore_index=True)
        
        st.success("✅ Data saved successfully!")
        st.rerun()

# Main dashboard
if len(st.session_state.health_data) > 0:
    df = st.session_state.health_data.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_sleep = df['sleep_hours'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>😴 Average Sleep</h3>
            <h2>{avg_sleep:.1f} hours</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_exercise = df['exercise_minutes'].sum()
        st.markdown(f"""
        <div class="metric-card">
            <h3>🏃‍♀️ Total Exercise</h3>
            <h2>{total_exercise} min</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_mood = df['mood_score'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>😊 Average Mood</h3>
            <h2>{avg_mood:.1f}/10</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_energy = df['energy_level'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>⚡ Average Energy</h3>
            <h2>{avg_energy:.1f}/10</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Interactive visualizations
    tab1, tab2, tab3 = st.tabs(["📈 Trends", "🔍 Correlations", "💡 Insights"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Sleep and energy trend
            fig_sleep = px.line(df, x='date', y=['sleep_hours', 'energy_level'],
                              title="Sleep Hours vs Energy Level Over Time")
            fig_sleep.update_layout(height=400)
            st.plotly_chart(fig_sleep, use_container_width=True)
        
        with col2:
            # Mood and exercise trend  
            fig_mood = px.bar(df, x='date', y='exercise_minutes', 
                             color='mood_score',
                             title="Exercise Duration Colored by Mood Score")
            fig_mood.update_layout(height=400)
            st.plotly_chart(fig_mood, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Sleep vs Energy correlation
            fig_corr1 = px.scatter(df, x='sleep_hours', y='energy_level',
                                  color='mood_score', size='exercise_minutes',
                                  title="Sleep vs Energy (size=exercise, color=mood)")
            st.plotly_chart(fig_corr1, use_container_width=True)
        
        with col2:
            # Exercise vs Mood correlation
            fig_corr2 = px.scatter(df, x='exercise_minutes', y='mood_score',
                                  color='sleep_quality',
                                  title="Exercise vs Mood (color=sleep quality)")
            st.plotly_chart(fig_corr2, use_container_width=True)
    
    with tab3:
        # AI-powered insights
        if len(df) >= 3:
            sleep_energy_corr = df['sleep_hours'].corr(df['energy_level'])
            exercise_mood_corr = df['exercise_minutes'].corr(df['mood_score'])
            
            st.markdown("""
            <div class="insight-box">
                <h3>🧠 AI-Powered Insights</h3>
            </div>
            """, unsafe_allow_html=True)
            
            if sleep_energy_corr > 0.5:
                st.success(f"💡 **Strong positive correlation** between sleep and energy ({sleep_energy_corr:.2f})! More sleep = more energy for you.")
            elif sleep_energy_corr < -0.3:
                st.warning(f"⚠️ **Negative correlation** detected ({sleep_energy_corr:.2f}). Something might be affecting your sleep quality.")
            
            if exercise_mood_corr > 0.3:
                st.success(f"🎯 **Exercise boosts your mood!** Correlation: {exercise_mood_corr:.2f}. Keep moving!")
            
            # Optimal sleep recommendation
            high_energy_days = df[df['energy_level'] >= 8]
            if len(high_energy_days) > 0:
                optimal_sleep = high_energy_days['sleep_hours'].mean()
                st.info(f"🎯 **Your optimal sleep duration**: {optimal_sleep:.1f} hours (based on your highest energy days)")
        else:
            st.info("📊 Add more data points to unlock AI-powered insights!")
    
    # Data export
    st.markdown("---")
    if st.button("📥 Export Data as CSV"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download health_data.csv",
            data=csv,
            file_name=f"health_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

else:
    st.info("👆 Start by logging your first day of health data in the sidebar!")
    
    # Demo data button
    if st.button("🎯 Load Demo Data (7 days)"):
        demo_dates = [datetime.now().date() - timedelta(days=i) for i in range(7)]
        demo_data = []
        
        for i, date in enumerate(demo_dates):
            demo_data.append({
                'date': date,
                'sleep_hours': 6.5 + np.random.normal(0, 1),
                'sleep_quality': np.random.randint(6, 9),
                'exercise_minutes': np.random.randint(20, 90),
                'exercise_type': np.random.choice(['Running', 'Gym', 'Yoga', 'Walking']),
                'mood_score': np.random.randint(6, 9),
                'energy_level': np.random.randint(5, 9),
                'water_glasses': np.random.randint(6, 12)
            })
        
        st.session_state.health_data = pd.DataFrame(demo_data)
        st.success("Demo data loaded! Refresh to see your dashboard.")
        st.rerun()

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit • Track daily, optimize for life 🌟*")