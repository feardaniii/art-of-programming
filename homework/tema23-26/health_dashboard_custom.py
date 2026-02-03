import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
st.set_page_config(
    page_title="🤖 AI Health Dashboard",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_FEATURES = [
    'sleep_hours', 'sleep_quality', 'exercise_minutes', 
    'mood_score', 'stress_level', 'water_intake', 
    'productive_hours', 'outdoor_time', 'day_of_week', 'is_weekend'
]

# --- Custom Styling ---
st.markdown("""
<style>
.stApp { background-color: #f8f9fa; }
.ai-metric {
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
    padding: 1.5rem; border-radius: 0.75rem; color: white;
    text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
.prediction-card {
    background: linear-gradient(135deg, #059669 0%, #10b981 100%);
    padding: 1.5rem; border-radius: 0.75rem; color: white;
    margin: 1rem 0; box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
.goal-card {
    background: white; padding: 1rem; border-radius: 0.5rem;
    border-left: 4px solid #3b82f6; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# --- Session State ---
if 'health_data' not in st.session_state:
    st.session_state.health_data = pd.DataFrame()
if 'wellness_model' not in st.session_state:
    st.session_state.wellness_model = None
if 'model_accuracy' not in st.session_state:
    st.session_state.model_accuracy = None
if 'quick_log_mode' not in st.session_state:
    st.session_state.quick_log_mode = False
if 'goals' not in st.session_state:
    st.session_state.goals = {
        'sleep_hours': 8.0, 'exercise_minutes': 30,
        'water_intake': 8, 'productive_hours': 6
    }

def calculate_wellness_score(row):
    sleep_norm = min(row['sleep_hours'] / 8, 1.0)
    sleep_quality_norm = row['sleep_quality'] / 10
    exercise_norm = min(row['exercise_minutes'] / 60, 1.0)
    mood_norm = row['mood_score'] / 10
    stress_norm = 1 - (row['stress_level'] / 10)
    water_norm = min(row['water_intake'] / 8, 1.0)
    productive_norm = min(row['productive_hours'] / 8, 1.0)
    
    wellness = (sleep_norm * 0.20 + sleep_quality_norm * 0.15 + 
                exercise_norm * 0.15 + mood_norm * 0.15 + 
                stress_norm * 0.15 + water_norm * 0.10 + 
                productive_norm * 0.10) * 10
    return round(wellness, 2)

# --- Sidebar ---
with st.sidebar:
    st.header("📝 Daily Health Log")
    
    quick_mode = st.checkbox("⚡ Quick Log Mode", value=st.session_state.quick_log_mode)
    st.session_state.quick_log_mode = quick_mode
    
    log_date = st.date_input("Date", datetime.now().date())
    
    if quick_mode:
        st.markdown("**Essential Metrics:**")
        sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.5, 0.5)
        exercise_minutes = st.slider("Exercise (min)", 0, 180, 30)
        mood_score = st.slider("Mood (1-10)", 1, 10, 7)
        water_intake = st.slider("Water (glasses)", 0, 15, 8)
        productive_hours = st.slider("Productive Hours", 0, 16, 6)
        sleep_quality = 7
        stress_level = 5
        outdoor_time = 30
    else:
        with st.expander("💤 Sleep & Rest", expanded=True):
            sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.5, 0.5)
            sleep_quality = st.slider("Sleep Quality (1-10)", 1, 10, 7)
        with st.expander("💪 Physical Activity", expanded=True):
            exercise_minutes = st.slider("Exercise (min)", 0, 180, 30)
            outdoor_time = st.slider("Outdoor Time (min)", 0, 240, 30)
        with st.expander("🧠 Mental & Productivity", expanded=True):
            mood_score = st.slider("Mood (1-10)", 1, 10, 7)
            stress_level = st.slider("Stress (1-10)", 1, 10, 4)
            productive_hours = st.slider("Productive Hours", 0, 16, 6)
        with st.expander("🥤 Nutrition", expanded=True):
            water_intake = st.slider("Water (glasses)", 0, 15, 8)
    
    if st.button("💾 Save Entry", type="primary", use_container_width=True):
        is_weekend = log_date.weekday() >= 5
        new_data = {
            'date': log_date, 'sleep_hours': sleep_hours,
            'sleep_quality': sleep_quality, 'exercise_minutes': exercise_minutes,
            'mood_score': mood_score, 'stress_level': stress_level,
            'water_intake': water_intake, 'productive_hours': productive_hours,
            'outdoor_time': outdoor_time, 'day_of_week': log_date.weekday(),
            'is_weekend': int(is_weekend)
        }
        new_data['wellness_score'] = calculate_wellness_score(pd.Series(new_data))
        st.session_state.health_data = pd.concat([
            st.session_state.health_data, pd.DataFrame([new_data])
        ], ignore_index=True)
        st.toast("✅ Data saved!", icon="✅")
    
    st.markdown("---")
    
    with st.expander("🎯 Goals", expanded=False):
        st.session_state.goals['sleep_hours'] = st.number_input(
            "Sleep Goal (hrs)", 4.0, 12.0, st.session_state.goals['sleep_hours'], 0.5)
        st.session_state.goals['exercise_minutes'] = st.number_input(
            "Exercise Goal (min)", 0, 180, st.session_state.goals['exercise_minutes'])
        st.session_state.goals['water_intake'] = st.number_input(
            "Water Goal", 0, 20, st.session_state.goals['water_intake'])
        st.session_state.goals['productive_hours'] = st.number_input(
            "Productive Goal", 0, 16, st.session_state.goals['productive_hours'])
    
    st.markdown("---")
    
    if len(st.session_state.health_data) > 0:
        csv = st.session_state.health_data.to_csv(index=False)
        st.download_button("📥 Export CSV", csv, 
                          f"health_data_{datetime.now().strftime('%Y%m%d')}.csv",
                          "text/csv", use_container_width=True)
    
    st.markdown("---")
    st.header("🤖 AI Model")
    
    if len(st.session_state.health_data) >= 10:
        if st.button("🚀 Train Model", use_container_width=True):
            with st.spinner("Training..."):
                try:
                    df = st.session_state.health_data.copy()
                    if 'wellness_score' not in df.columns:
                        df['wellness_score'] = df.apply(calculate_wellness_score, axis=1)
                    df['is_weekend'] = df['is_weekend'].astype(int)
                    
                    X = df[MODEL_FEATURES]
                    y = df['wellness_score']
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42)
                    
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    
                    y_pred = model.predict(X_test)
                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    st.session_state.wellness_model = model
                    st.session_state.model_accuracy = {
                        'r2': r2, 'mae': mae,
                        'feature_importance': dict(zip(MODEL_FEATURES, model.feature_importances_))
                    }
                    st.success(f"✅ Trained! R²: {r2:.2%}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.info(f"Need {10 - len(st.session_state.health_data)} more entries")



# --- Main Dashboard ---
st.title("🤖 AI-Powered Health Dashboard")

if len(st.session_state.health_data) > 0:
    df = st.session_state.health_data.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    if 'wellness_score' not in df.columns:
        df['wellness_score'] = df.apply(calculate_wellness_score, axis=1)
    
    # --- Top Metrics Row ---
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_wellness = df['wellness_score'].mean()
        st.metric("📊 Avg Wellness", f"{avg_wellness:.1f}/10",
                 delta=f"{df['wellness_score'].iloc[-1] - avg_wellness:+.1f}")
    
    with col2:
        avg_sleep = df['sleep_hours'].mean()
        delta_sleep = avg_sleep - st.session_state.goals['sleep_hours']
        st.metric("😴 Avg Sleep", f"{avg_sleep:.1f}hrs",
                 delta=f"{delta_sleep:+.1f} vs goal")
    
    with col3:
        total_exercise = df['exercise_minutes'].sum()
        st.metric("💪 Total Exercise", f"{total_exercise} min",
                 delta=f"{len(df)} days")
    
    with col4:
        avg_water = df['water_intake'].mean()
        st.metric("💧 Avg Water", f"{avg_water:.1f} glasses",
                 delta=f"{avg_water - st.session_state.goals['water_intake']:+.1f}")
    
    st.markdown("---")
    
    # --- Goal Progress ---
    st.subheader("🎯 Goal Tracking")
    
    goal_col1, goal_col2, goal_col3, goal_col4 = st.columns(4)
    latest = df.iloc[-1]
    
    with goal_col1:
        sleep_prog = (latest['sleep_hours'] / st.session_state.goals['sleep_hours']) * 100
        st.markdown(f"""
        <div class="goal-card">
            <strong>Sleep Goal</strong><br>
            {latest['sleep_hours']:.1f} / {st.session_state.goals['sleep_hours']} hrs
        </div>
        """, unsafe_allow_html=True)
        st.progress(min(sleep_prog / 100, 1.0))
    
    with goal_col2:
        ex_prog = (latest['exercise_minutes'] / st.session_state.goals['exercise_minutes']) * 100
        st.markdown(f"""
        <div class="goal-card">
            <strong>Exercise Goal</strong><br>
            {latest['exercise_minutes']} / {st.session_state.goals['exercise_minutes']} min
        </div>
        """, unsafe_allow_html=True)
        st.progress(min(ex_prog / 100, 1.0))
    
    with goal_col3:
        water_prog = (latest['water_intake'] / st.session_state.goals['water_intake']) * 100
        st.markdown(f"""
        <div class="goal-card">
            <strong>Water Goal</strong><br>
            {latest['water_intake']} / {st.session_state.goals['water_intake']} glasses
        </div>
        """, unsafe_allow_html=True)
        st.progress(min(water_prog / 100, 1.0))
    
    with goal_col4:
        prod_prog = (latest['productive_hours'] / st.session_state.goals['productive_hours']) * 100
        st.markdown(f"""
        <div class="goal-card">
            <strong>Productivity Goal</strong><br>
            {latest['productive_hours']} / {st.session_state.goals['productive_hours']} hrs
        </div>
        """, unsafe_allow_html=True)
        st.progress(min(prod_prog / 100, 1.0))
    
    st.markdown("---")
    
    # --- Prediction Section ---
    if st.session_state.wellness_model is not None:
        st.markdown("""
        <div class="prediction-card">
            <h3>🔮 Tomorrow's Wellness Predictor</h3>
            <p>Predict your wellness score based on planned activities</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1.2, 1, 1])
        
        with col1:
            st.subheader("🎛️ Simulator")
            
            avg_quality = float(df['sleep_quality'].mean())
            avg_outdoor = float(df['outdoor_time'].mean())
            
            p_sleep = st.slider("Planned Sleep (hrs)", 4.0, 12.0, 8.0, key="p_sleep")
            p_exer = st.slider("Planned Exercise (min)", 0, 120, 30, key="p_exer")
            p_water = st.slider("Planned Water", 0, 15, 8, key="p_water")
            p_productive = st.slider("Planned Productive Hrs", 0, 12, 6, key="p_prod")
            
            with st.expander("Advanced", expanded=False):
                p_mood = st.slider("Expected Mood", 1, 10, 7, key="p_mood")
                p_stress = st.slider("Expected Stress", 1, 10, 5, key="p_stress")
            
            tomorrow = datetime.now().date() + timedelta(days=1)
            is_wknd = 1 if tomorrow.weekday() >= 5 else 0
            
            input_data = pd.DataFrame([{
                'sleep_hours': p_sleep, 'sleep_quality': avg_quality,
                'exercise_minutes': p_exer, 'mood_score': p_mood,
                'stress_level': p_stress, 'water_intake': p_water,
                'productive_hours': p_productive, 'outdoor_time': avg_outdoor,
                'day_of_week': tomorrow.weekday(), 'is_weekend': is_wknd
            }])
            
            input_vector = input_data[MODEL_FEATURES]
            
            try:
                pred_wellness = st.session_state.wellness_model.predict(input_vector)[0]
                
                if pred_wellness >= 8:
                    color, emoji, label = "#059669", "🌟", "Excellent"
                elif pred_wellness >= 6:
                    color, emoji, label = "#10b981", "😊", "Good"
                elif pred_wellness >= 4:
                    color, emoji, label = "#f59e0b", "😐", "Fair"
                else:
                    color, emoji, label = "#ef4444", "😟", "Needs Work"
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {color} 0%, {color}dd 100%); 
                            padding: 25px; border-radius: 12px; color: white; text-align: center; 
                            margin-top: 15px; box-shadow: 0 6px 16px rgba(0,0,0,0.2);">
                    <div style="font-size: 3rem; margin-bottom: 10px;">{emoji}</div>
                    <div style="font-size: 1.3rem;">Predicted Wellness</div>
                    <div style="font-size: 3.5rem; font-weight: bold; margin: 10px 0;">{pred_wellness:.1f}/10</div>
                    <div style="font-size: 1.2rem; opacity: 0.9;">{label}</div>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Prediction Error: {e}")

        with col2:
            st.subheader("📊 Accuracy")
            acc = st.session_state.model_accuracy
            st.metric("Model Confidence (R²)", f"{acc['r2']:.1%}")
            st.metric("Avg Error (MAE)", f"{acc['mae']:.2f} pts")
            st.info("💡 Model predicts overall wellness (1-10)")

        with col3:
            st.subheader("🔑 Top Factors")
            imp = pd.DataFrame(
                list(acc['feature_importance'].items()), 
                columns=['Feature', 'Importance']
            ).sort_values('Importance', ascending=True).tail(5)
            
            imp['Feature'] = imp['Feature'].str.replace('_', ' ').str.title()
            
            fig = px.bar(imp, x='Importance', y='Feature', orientation='h', 
                         color='Importance', 
                         color_continuous_scale=['#1e3a8a', '#3b82f6', '#60a5fa'])
            fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0), 
                             showlegend=False, paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")        



# --- Tabs Section ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Trend Analysis", 
        "📊 Weekly Summary", 
        "🌡️ Correlations", 
        "🧪 Demo Data"
    ])
    
    with tab1:
        st.subheader("Multi-Metric Trend (Sleep, Exercise, Wellness)")
        
        # Animated multi-metric chart
        fig_multi = go.Figure()
        
        # Sleep trace
        fig_multi.add_trace(go.Scatter(
            x=df['date'], y=df['sleep_hours'], 
            name='Sleep Hours',
            line=dict(color='#3b82f6', width=3),
            mode='lines+markers',
            marker=dict(size=8, line=dict(width=2, color='white'))
        ))
        
        # Exercise trace (scaled)
        fig_multi.add_trace(go.Scatter(
            x=df['date'], y=df['exercise_minutes'] / 10,
            name='Exercise (×10 min)',
            line=dict(color='#10b981', width=3, dash='dot'),
            mode='lines+markers',
            marker=dict(size=8, symbol='square', line=dict(width=2, color='white')),
            customdata=df['exercise_minutes']
        ))
        
        # Wellness trace (right axis)
        fig_multi.add_trace(go.Scatter(
            x=df['date'], y=df['wellness_score'], 
            name='Wellness Score',
            line=dict(color='#f59e0b', width=4),
            mode='lines+markers',
            marker=dict(size=10, symbol='diamond', line=dict(width=2, color='white')),
            yaxis='y2'
        ))
        
        fig_multi.update_layout(
            title="Key Metrics Over Time",
            xaxis=dict(title="Date"),
            yaxis=dict(title="Sleep (hrs) / Exercise (×10 min)"),
            yaxis2=dict(title="Wellness Score", overlaying='y', side='right', range=[0, 10]),
            height=450,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        
        st.plotly_chart(fig_multi, use_container_width=True)
    
    with tab2:
        st.subheader("📊 Weekly Averages Comparison")
        
        # Calculate weekly stats
        df['week'] = df['date'].dt.isocalendar().week
        df['year'] = df['date'].dt.year
        df['week_label'] = df['year'].astype(str) + '-W' + df['week'].astype(str).str.zfill(2)
        
        weekly_stats = df.groupby('week_label').agg({
            'sleep_hours': 'mean',
            'exercise_minutes': 'mean',
            'wellness_score': 'mean',
            'water_intake': 'mean',
            'productive_hours': 'mean',
            'mood_score': 'mean'
        }).round(2)
        
        # Grouped bar chart
        fig_weekly = go.Figure()
        
        metrics_info = [
            ('sleep_hours', '#3b82f6', 'Sleep (hrs)', 1.2),
            ('exercise_minutes', '#10b981', 'Exercise (min)', 18),
            ('wellness_score', '#f59e0b', 'Wellness', 1),
            ('water_intake', '#06b6d4', 'Water (glasses)', 1.5),
            ('productive_hours', '#8b5cf6', 'Productive (hrs)', 1.2),
            ('mood_score', '#ec4899', 'Mood', 1)
        ]
        
        for metric, color, name, scale in metrics_info:
            normalized = weekly_stats[metric] / scale
            
            fig_weekly.add_trace(go.Bar(
                x=weekly_stats.index,
                y=normalized,
                name=name,
                marker_color=color,
                customdata=weekly_stats[metric].round(1)
            ))
        
        fig_weekly.update_layout(
            title="Weekly Averages (Normalized 0-10)",
            xaxis_title="Week",
            yaxis_title="Normalized Score",
            barmode='group',
            height=400,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        
        st.plotly_chart(fig_weekly, use_container_width=True)
        
        st.markdown("**Detailed Weekly Statistics:**")
        st.dataframe(weekly_stats.style.background_gradient(cmap='Blues', axis=0),
                    use_container_width=True)
    
    with tab3:
        st.subheader("🌡️ Correlation Matrix")
        
        numeric_cols = ['sleep_hours', 'sleep_quality', 'exercise_minutes', 
                       'mood_score', 'stress_level', 'water_intake', 
                       'productive_hours', 'wellness_score']
        
        corr = df[numeric_cols].corr()
        
        fig_corr = px.imshow(
            corr, text_auto=".2f", 
            color_continuous_scale="RdBu_r",
            aspect="auto",
            title="Correlation Analysis"
        )
        fig_corr.update_layout(height=500, xaxis=dict(tickangle=-45))
        
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.markdown("**🔍 Strongest Correlations:**")
        
        corr_flat = corr.where(np.triu(np.ones_like(corr, dtype=bool), k=1))
        strong = corr_flat.unstack().sort_values(ascending=False).head(3)
        
        for (v1, v2), val in strong.items():
            if pd.notna(val):
                st.info(f"**{v1.replace('_', ' ').title()}** ↔ **{v2.replace('_', ' ').title()}**: {val:.2f}")
    
    with tab4:
        st.markdown("### 🧪 Load Demo Data")
        
        st.info("Load 30 days of synthetic health data to test features and train the model.")
        
        if st.button("📥 Load 30 Days Demo Data", type="primary"):
            demo_data = []
            today = datetime.now().date()
            
            for i in range(30):
                d = today - timedelta(days=30-i)
                is_wknd = d.weekday() >= 5
                
                sleep = np.random.normal(8 if is_wknd else 7, 1)
                exercise = np.random.normal(45 if is_wknd else 20, 15)
                water = np.random.normal(8, 2)
                productive = np.random.normal(7 if not is_wknd else 3, 2)
                
                entry = {
                    'date': d,
                    'sleep_hours': max(4, min(12, sleep)),
                    'sleep_quality': np.random.randint(5, 10),
                    'exercise_minutes': max(0, int(exercise)),
                    'mood_score': np.random.randint(4, 10),
                    'stress_level': np.random.randint(2, 8),
                    'water_intake': max(0, min(15, int(water))),
                    'productive_hours': max(0, min(16, int(productive))),
                    'outdoor_time': np.random.randint(0, 120),
                    'day_of_week': d.weekday(),
                    'is_weekend': int(is_wknd)
                }
                entry['wellness_score'] = calculate_wellness_score(pd.Series(entry))
                demo_data.append(entry)
            
            st.session_state.health_data = pd.DataFrame(demo_data)
            st.success("✅ Demo data loaded! Train model in sidebar.")
            st.rerun()

else:
    # Empty state
    st.info("👈 **Get Started:** Log your first entry or load demo data!")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; 
                    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); 
                    border-radius: 1rem; color: white; margin: 2rem 0;">
            <h2>Welcome to Your Health Dashboard</h2>
            <p>Track wellness, predict outcomes, achieve goals with AI.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📥 Load Demo Data", type="primary", use_container_width=True):
            demo_data = []
            today = datetime.now().date()
            
            for i in range(30):
                d = today - timedelta(days=30-i)
                is_wknd = d.weekday() >= 5
                sleep = np.random.normal(8 if is_wknd else 7, 1)
                exercise = np.random.normal(45 if is_wknd else 20, 15)
                water = np.random.normal(8, 2)
                productive = np.random.normal(7 if not is_wknd else 3, 2)
                
                entry = {
                    'date': d,
                    'sleep_hours': max(4, min(12, sleep)),
                    'sleep_quality': np.random.randint(5, 10),
                    'exercise_minutes': max(0, int(exercise)),
                    'mood_score': np.random.randint(4, 10),
                    'stress_level': np.random.randint(2, 8),
                    'water_intake': max(0, min(15, int(water))),
                    'productive_hours': max(0, min(16, int(productive))),
                    'outdoor_time': np.random.randint(0, 120),
                    'day_of_week': d.weekday(),
                    'is_weekend': int(is_wknd)
                }
                entry['wellness_score'] = calculate_wellness_score(pd.Series(entry))
                demo_data.append(entry)
            
            st.session_state.health_data = pd.DataFrame(demo_data)
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.9rem;">
    🤖 AI Health Dashboard | Streamlit + Plotly + Scikit-learn
</div>
""", unsafe_allow_html=True)