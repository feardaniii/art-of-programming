import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# PAGE CONFIG
st.set_page_config(page_title="Evolution of Data Apps", page_icon="🚀")

# ===================================================
# 1. THE "BEFORE" (STATIC)
# We display this as code so students can compare, 
# but we don't run it because plt.show() blocks web apps.
# ===================================================
st.header("1. The Old Way: Static Scripts 🐢")
st.markdown("This code runs once, produces one image, and stops. To change inputs, you have to edit the code.")

st.code("""
# ❌ STATIC ANALYSIS
import pandas as pd
import matplotlib.pyplot as plt

def analyze_health_data():
    # Hardcoded data loading
    df = pd.read_csv('health_data.csv') 
    correlation = df['sleep_hours'].corr(df['energy_level'])
    
    # Static image generation
    plt.figure(figsize=(10, 6))
    plt.scatter(df['sleep_hours'], df['energy_level'])
    plt.title(f'Sleep vs Energy (r={correlation:.2f})')
    plt.show() # Blocks execution!
    
    print(f"Correlation: {correlation:.2f}")

analyze_health_data()
""", language='python')

st.markdown("---")

# ===================================================
# 2. THE "AFTER" (INTERACTIVE)
# ===================================================
st.header("2. The New Way: Interactive Apps 🚀")
st.markdown("*Real-time insights for better decisions*")

# --- INITIALIZE SESSION STATE (Fixing the missing logic) ---
if 'health_data' not in st.session_state:
    # Initialize with some dummy data so the chart isn't empty on first load
    st.session_state.health_data = pd.DataFrame({
        'sleep_hours': [5.0, 6.0, 7.5, 8.0, 9.0],
        'energy_level': [4, 5, 8, 9, 8]
    })

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.header("📝 Log Today's Data")
    
    # Input widgets
    sleep_input = st.slider("Sleep Hours", 0.0, 12.0, 7.0, step=0.5)
    energy_input = st.slider("Energy Level", 1, 10, 6)
    
    # The Button Logic
    if st.button("💾 Save Data", type="primary"):
        # Create a new row
        new_data = pd.DataFrame({
            'sleep_hours': [sleep_input],
            'energy_level': [energy_input]
        })
        
        # Append to session state
        st.session_state.health_data = pd.concat(
            [st.session_state.health_data, new_data], 
            ignore_index=True
        )
        
        st.success("✅ Data saved!")
        # Rerun to update charts immediately
        st.rerun()

# --- MAIN DASHBOARD ---

# Get data from state
df = st.session_state.health_data

col1, col2 = st.columns([2, 1])

with col1:
    # Real-time visualization
    # Note: trendline='ols' requires 'pip install statsmodels'
    try:
        fig = px.scatter(df, x='sleep_hours', y='energy_level', 
                        title='Sleep vs Energy - Your Personal Pattern',
                        trendline='ols',
                        size='energy_level',
                        color='energy_level')
    except:
        # Fallback if statsmodels is not installed
        fig = px.scatter(df, x='sleep_hours', y='energy_level', 
                        title='Sleep vs Energy - Your Personal Pattern',
                        size='energy_level')
        
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🤖 AI Insights")
    
    # Dynamic calculations
    if len(df) > 1:
        correlation = df['sleep_hours'].corr(df['energy_level'])
        
        st.metric("Correlation", f"{correlation:.2f}")
        
        if correlation > 0.5:
            st.success("💡 **Insight:** Strong link detected! More sleep directly boosts your energy.")
        elif correlation < -0.5:
            st.warning("💡 **Insight:** Odd pattern detected. More sleep seems to lower energy?")
        else:
            st.info("💡 **Insight:** No clear pattern yet. Keep logging data!")
            
        # Actionable recommendation
        high_energy_days = df[df['energy_level'] >= 8]
        if not high_energy_days.empty:
            optimal_sleep = high_energy_days['sleep_hours'].mean()
            st.markdown(f"""
            ### 🎯 Recommendation
            To feel your best, aim for:
            **{optimal_sleep:.1f} hours**
            """)
    else:
        st.info("Log more data to see insights!")

# --- FOOTER ---
st.markdown("---")
st.markdown("""
### 🏆 Why this wins:
1.  **Interactive:** Users change inputs, results update instantly.
2.  **Persistent:** Data accumulates in `session_state` (temporarily).
3.  **Actionable:** It doesn't just show a graph; it calculates an "Optimal Target."
""")