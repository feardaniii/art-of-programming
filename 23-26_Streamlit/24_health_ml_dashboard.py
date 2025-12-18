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

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
st.set_page_config(
    page_title="🤖 AI Health Dashboard",
    page_icon="hrt",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define the exact features used for the Model to ensure consistency
MODEL_FEATURES = [
    'sleep_hours', 'sleep_quality', 'exercise_minutes', 
    'mood_score', 'stress_level', 'caffeine_cups', 
    'screen_time', 'outdoor_time', 'day_of_week', 'is_weekend'
]

# --- Custom Styling ---
st.markdown("""
<style>
.ai-metric {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 0.75rem;
    color: white;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
.prediction-card {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    padding: 1.5rem;
    border-radius: 0.75rem;
    color: white;
    margin: 1rem 0;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
.ai-insight {
    background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    padding: 1rem;
    border-radius: 0.5rem;
    color: #2c3e50;
    font-weight: 500;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if 'health_data' not in st.session_state:
    st.session_state.health_data = pd.DataFrame()
if 'energy_model' not in st.session_state:
    st.session_state.energy_model = None
if 'model_accuracy' not in st.session_state:
    st.session_state.model_accuracy = None

# --- Sidebar: Data Input ---
with st.sidebar:
    st.header("📝 Daily Log")
    
    log_date = st.date_input("Date", datetime.now().date())
    
    with st.expander("😴 Sleep & Activity", expanded=True):
        sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.5, 0.5)
        sleep_quality = st.slider("Sleep Quality (1-10)", 1, 10, 7)
        exercise_minutes = st.slider("Exercise (min)", 0, 180, 30)
    
    with st.expander("🧠 Mental State", expanded=True):
        mood_score = st.slider("Mood (1-10)", 1, 10, 7)
        stress_level = st.slider("Stress (1-10)", 1, 10, 4)
        energy_level = st.slider("Energy (1-10)", 1, 10, 7)
    
    with st.expander("🔬 Other Factors", expanded=False):
        caffeine_cups = st.slider("Caffeine (cups)", 0, 8, 2)
        screen_time = st.slider("Screen Time (hours)", 0, 16, 6)
        outdoor_time = st.slider("Outdoor Time (min)", 0, 240, 30)
    
    if st.button("💾 Save Entry", type="primary"):
        # Calculate derived features
        is_weekend = log_date.weekday() >= 5
        
        new_data = {
            'date': log_date,
            'sleep_hours': sleep_hours,
            'sleep_quality': sleep_quality,
            'exercise_minutes': exercise_minutes,
            'mood_score': mood_score,
            'stress_level': stress_level,
            'energy_level': energy_level,
            'caffeine_cups': caffeine_cups,
            'screen_time': screen_time,
            'outdoor_time': outdoor_time,
            'day_of_week': log_date.weekday(),
            'is_weekend': int(is_weekend) # Store as int (0 or 1) for ML stability
        }
        
        # Append data
        st.session_state.health_data = pd.concat([
            st.session_state.health_data,
            pd.DataFrame([new_data])
        ], ignore_index=True)
        
        st.toast("Data saved successfully!", icon="✅")

    st.markdown("---")
    
    # --- Sidebar: Model Training ---
    st.header("🤖 AI Settings")
    
    data_count = len(st.session_state.health_data)
    
    if data_count >= 10:
        if st.button("🚀 Train New Model", type="secondary"):
            with st.spinner("Training Random Forest Model..."):
                try:
                    df = st.session_state.health_data.copy()
                    
                    # Ensure correct types
                    df['is_weekend'] = df['is_weekend'].astype(int)
                    
                    X = df[MODEL_FEATURES]
                    y = df['energy_level']
                    
                    # Split & Train
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                    
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    
                    # Metrics
                    y_pred = model.predict(X_test)
                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    st.session_state.energy_model = model
                    st.session_state.model_accuracy = {
                        'r2': r2,
                        'mae': mae,
                        'feature_importance': dict(zip(MODEL_FEATURES, model.feature_importances_))
                    }
                    st.success(f"Trained! Accuracy: {r2:.2f}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")
    else:
        st.warning(f"Need {10 - data_count} more entries to train.")

# --- Main Dashboard ---
st.title("🤖 AI-Powered Health Dashboard")

if len(st.session_state.health_data) > 0:
    df = st.session_state.health_data.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # --- Prediction Section ---
    if st.session_state.energy_model is not None:
        st.markdown("""
        <div class="prediction-card">
            <h3>🔮 Tomorrow's Energy Predictor</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1.2, 1, 1])
        
        with col1:
            st.subheader("🎛️ Simulator")
            # Calculate averages to use as default values
            avg_quality = float(df['sleep_quality'].mean())
            avg_screen = float(df['screen_time'].mean())
            avg_outdoor = float(df['outdoor_time'].mean())
            
            p_sleep = st.slider("Planned Sleep (hrs)", 4.0, 12.0, 8.0, key="p_sleep")
            p_exer = st.slider("Planned Exercise (min)", 0, 120, 30, key="p_exer")
            p_caff = st.slider("Planned Caffeine (cups)", 0, 6, 2, key="p_caff")
            
            with st.expander("Advanced Inputs (Mood/Stress)", expanded=False):
                p_mood = st.slider("Expected Mood", 1, 10, 7, key="p_mood")
                p_stress = st.slider("Expected Stress", 1, 10, 5, key="p_stress")
            
            # Prepare Input Vector
            tomorrow = datetime.now().date() + timedelta(days=1)
            is_wknd = 1 if tomorrow.weekday() >= 5 else 0
            
            # Map inputs strictly to MODEL_FEATURES list
            input_data = pd.DataFrame([{
                'sleep_hours': p_sleep,
                'sleep_quality': avg_quality, # Using historical average
                'exercise_minutes': p_exer,
                'mood_score': p_mood,
                'stress_level': p_stress,
                'caffeine_cups': p_caff,
                'screen_time': avg_screen,    # Using historical average
                'outdoor_time': avg_outdoor,  # Using historical average
                'day_of_week': tomorrow.weekday(),
                'is_weekend': is_wknd
            }])
            
            # Ensure column order matches training
            input_vector = input_data[MODEL_FEATURES]
            
            try:
                predicted_val = st.session_state.energy_model.predict(input_vector)[0]
                
                # Visual Feedback
                color = "#48bb78" if predicted_val > 7 else "#ecc94b" if predicted_val > 5 else "#f56565"
                st.markdown(f"""
                <div style="background-color: {color}; padding: 20px; border-radius: 10px; color: white; text-align: center; margin-top: 10px;">
                    <div style="font-size: 1.2rem;">Predicted Energy</div>
                    <div style="font-size: 3rem; font-weight: bold;">{predicted_val:.1f}/10</div>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Prediction Error: {e}")

        with col2:
            st.subheader("📊 Accuracy")
            acc = st.session_state.model_accuracy
            st.metric("Model Confidence (R²)", f"{acc['r2']:.1%}")
            st.metric("Avg Error (MAE)", f"{acc['mae']:.2f}")
            
            st.info("💡 The model uses your historical averages for Sleep Quality and Screen Time to make this prediction realistic.")

        with col3:
            st.subheader("🔑 Key Drivers")
            imp = pd.DataFrame(
                list(acc['feature_importance'].items()), 
                columns=['Feature', 'Importance']
            ).sort_values('Importance', ascending=True).tail(5)
            
            fig = px.bar(imp, x='Importance', y='Feature', orientation='h', 
                         color='Importance', color_continuous_scale='Viridis')
            fig.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    
    # --- Tabs Section ---
    tab1, tab2, tab3 = st.tabs(["📈 Data Trends", "🌡️ Correlations", "🧪 Demo Data"])
    
    with tab1:
        # Dual axis chart for sleep vs energy
        fig_dual = go.Figure()
        fig_dual.add_trace(go.Bar(x=df['date'], y=df['exercise_minutes'], name='Exercise (min)', marker_color='#a3bffa', opacity=0.6))
        fig_dual.add_trace(go.Scatter(x=df['date'], y=df['energy_level'], name='Energy Level', line=dict(color='#667eea', width=3), yaxis='y2'))
        
        fig_dual.update_layout(
            title="Exercise Volume vs Energy Levels",
            yaxis=dict(title="Exercise Minutes"),
            yaxis2=dict(title="Energy Level", overlaying='y', side='right'),
            height=400,
            hovermode="x unified"
        )
        st.plotly_chart(fig_dual, use_container_width=True)
    
    with tab2:
        # Clean numeric data for heatmap
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto",
                             title="What is correlated with what?")
        st.plotly_chart(fig_corr, use_container_width=True)

    with tab3:
        st.markdown("### Need Data?")
        if st.button("📥 Load 30 Days of Demo Data"):
            demo_data = []
            today = datetime.now().date()
            for i in range(30):
                d = today - timedelta(days=30-i)
                is_wknd = d.weekday() >= 5
                
                # Generate realistic correlated data
                sleep = np.random.normal(8 if is_wknd else 7, 1)
                exercise = np.random.normal(45 if is_wknd else 20, 15)
                # Energy calculation logic for "ground truth"
                energy = (sleep * 0.5) + (exercise * 0.05) + np.random.normal(0, 1)
                energy = max(1, min(10, energy))
                
                demo_data.append({
                    'date': d,
                    'sleep_hours': max(4, min(12, sleep)),
                    'sleep_quality': np.random.randint(5, 10),
                    'exercise_minutes': max(0, int(exercise)),
                    'mood_score': np.random.randint(4, 10),
                    'stress_level': np.random.randint(2, 8),
                    'energy_level': energy,
                    'caffeine_cups': np.random.randint(1, 5),
                    'screen_time': np.random.uniform(2, 10),
                    'outdoor_time': np.random.randint(0, 120),
                    'day_of_week': d.weekday(),
                    'is_weekend': int(is_wknd)
                })
            
            st.session_state.health_data = pd.DataFrame(demo_data)
            st.success("Demo data loaded! Go to the sidebar and click 'Train New Model'.")
            st.rerun()

else:
    st.info("👈 Please enter your first data point in the sidebar or Load Demo Data in the tabs below.")
    if st.button("📥 Quick Load Demo Data"):
        # Quick load logic (same as above, just accessible when empty)
        demo_data = []
        today = datetime.now().date()
        for i in range(30):
            d = today - timedelta(days=30-i)
            is_wknd = d.weekday() >= 5
            sleep = np.random.normal(8 if is_wknd else 7, 1)
            exercise = np.random.normal(45 if is_wknd else 20, 15)
            energy = (sleep * 0.5) + (exercise * 0.05) + np.random.normal(0, 1)
            
            demo_data.append({
                'date': d,
                'sleep_hours': max(4, min(12, sleep)),
                'sleep_quality': np.random.randint(5, 10),
                'exercise_minutes': max(0, int(exercise)),
                'mood_score': np.random.randint(4, 10),
                'stress_level': np.random.randint(2, 8),
                'energy_level': max(1, min(10, energy)),
                'caffeine_cups': np.random.randint(1, 5),
                'screen_time': np.random.uniform(2, 10),
                'outdoor_time': np.random.randint(0, 120),
                'day_of_week': d.weekday(),
                'is_weekend': int(is_wknd)
            })
        st.session_state.health_data = pd.DataFrame(demo_data)
        st.rerun()