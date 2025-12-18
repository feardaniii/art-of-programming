import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import streamlit_authenticator as stauth
import sqlite3 
import time
import yaml
from yaml.loader import SafeLoader

# Notite Personale Detaliale 

# Aici setam configurarea paginii - 
# --- Page Configuration ---
st.set_page_config(
    page_title="👨‍👩‍👧‍👦 Family Dashboard V2.0",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Database Management ---
@st.cache_resource
def get_db_connection():
    """Initialize and return database connection (Cached Resource)"""
    conn = sqlite3.connect('family_dashboard.db', check_same_thread=False)
    
    # Enable row factory to access columns by name
    conn.row_factory = sqlite3.Row
    
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS health_data (
            id INTEGER PRIMARY KEY,
            user_id TEXT,
            date DATE,
            sleep_hours REAL,
            exercise_minutes INTEGER,
            mood_score INTEGER,
            energy_level INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS finance_data (
            id INTEGER PRIMARY KEY,
            user_id TEXT,
            date DATE,
            amount REAL,
            category TEXT,
            description TEXT,
            type TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS family_goals (
            id INTEGER PRIMARY KEY,
            goal_type TEXT,
            goal_name TEXT,
            target_value REAL,
            current_value REAL DEFAULT 0,
            start_date DATE,
            end_date DATE,
            created_by TEXT,
            status TEXT DEFAULT 'active'
        )
    ''')
    
    conn.commit()
    return conn

# --- Authentication Config ---
def get_auth_config():
    """Returns the configuration dictionary for authentication"""
    return {
        'credentials': {
            'usernames': {
                'dad': {
                    'name': 'Dad',
                    # Hash for 'password123'
                    'password': '111www',
                    'email': 'dad@family.com'
                },
                'mom': {
                    'name': 'Mom', 
                    'password': '111www',
                    'email': 'mom@family.com'
                },
                'teen': {
                    'name': 'Teen',
                    'password': '$2b$12$kQRYOtvHPPAU3dHjdJMz0e2oKjmEwKHR3wD7RgJ6LKJzPJ8J3tNzK',
                    'email': 'teen@family.com'
                }
            }
        },
        'cookie': {
            'name': 'family_dash_cookie',
            'key': 'some_random_secret_key',
            'expiry_days': 30
        }
    }

# --- Custom Styling ---
st.markdown("""
<style>
.family-card {
    background: white;
    padding: 1.5rem;
    border-radius: 0.75rem;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    text-align: center;
    border: 1px solid #f0f2f6;
}
.metric-value {
    font-size: 2rem;
    font-weight: bold;
    color: #1f77b4;
}
.metric-label {
    color: #666;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.activity-item {
    padding: 10px;
    border-bottom: 1px solid #eee;
    font-size: 0.9rem;
}
.activity-icon {
    margin-right: 10px;
    font-size: 1.2rem;
}
</style>
""", unsafe_allow_html=True)

# --- Main App Logic ---
def main():
    config = get_auth_config()
    
    # 1. Initialize Authenticator (FIXED: Removed preauthorized)
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )

    # 2. Login Widget
    try:
        # 'login' returns a tuple in newer versions or just handles session state
        authenticator.login()
    except Exception as e:
        st.error(f"Authentication Error: {e}")

    # 3. Check Authentication State
    if st.session_state["authentication_status"]:
        
        # Get User Info
        username = st.session_state["username"]
        name = config['credentials']['usernames'][username]['name']
        conn = get_db_connection()
        
        # --- Sidebar ---
        with st.sidebar:
            st.title(f"👤 {name}")
            authenticator.logout('Logout', 'main')
            st.markdown("---")
            
            # --- Quick Log Input (Moved to Sidebar for cleanliness) ---
            st.header("📝 Quick Log")
            log_type = st.selectbox("Log Type", ["Health", "Finance"])
            
            if log_type == "Health":
                with st.form("sidebar_health"):
                    s_sleep = st.number_input("Sleep (hrs)", 0.0, 12.0, 7.5, 0.5)
                    s_exercise = st.number_input("Exercise (min)", 0, 180, 30)
                    s_mood = st.slider("Mood", 1, 10, 7)
                    if st.form_submit_button("Save Health"):
                        conn.execute('INSERT INTO health_data (user_id, date, sleep_hours, exercise_minutes, mood_score) VALUES (?, date("now"), ?, ?, ?)', 
                                   (username, s_sleep, s_exercise, s_mood))
                        conn.commit()
                        st.toast("Health logged!", icon="🏃")
                        time.sleep(0.5)
                        st.rerun()

            elif log_type == "Finance":
                with st.form("sidebar_finance"):
                    f_amount = st.number_input("Amount (€)", 0.0, 1000.0, 10.0)
                    f_type = st.selectbox("Type", ["Expense", "Income"])
                    f_cat = st.selectbox("Category", ["Food", "Transport", "Bills", "Fun", "Other"])
                    f_desc = st.text_input("Note")
                    if st.form_submit_button("Save Money"):
                        # Store expenses as negative for math, but handle display logic later
                        final_amt = -f_amount if f_type == "Expense" else f_amount
                        conn.execute('INSERT INTO finance_data (user_id, date, amount, category, description, type) VALUES (?, date("now"), ?, ?, ?, ?)', 
                                   (username, final_amt, f_cat, f_desc, f_type))
                        conn.commit()
                        st.toast("Transaction logged!", icon="💰")
                        time.sleep(0.5)
                        st.rerun()

        # --- Dashboard Header ---
        col1, col2 = st.columns([6, 1])
        with col1:
            st.title("🏡 Family Dashboard TEMA")
            st.caption(f"Real-time overview for the **{username.capitalize()}** family")
        with col2:
            if st.button("🔄 Refresh"):
                st.rerun()


        # Această interogare SQL combină datele din două tabele diferite
        # (finance_data și health_data) într-un singur "feed" de activități.
        # SQL este ca un limbaj specializat pentru a prelua date din baze de date,
        # similar cu cum pandas preia date din CSV-uri, dar optimizat pentru volume mari.
        # --- Recent Activity Feed (Top Section) ---
        cursor = conn.cursor()
        cursor.execute('''
            SELECT user_id, type, amount, category, exercise_minutes, created_at 
            FROM (
                SELECT user_id, 'finance' as type, amount, category, 0 as exercise_minutes, created_at FROM finance_data
                UNION ALL
                SELECT user_id, 'health' as type, 0 as amount, 'Health' as category, exercise_minutes, created_at FROM health_data
            )
            ORDER BY created_at DESC LIMIT 5
        ''')
        activities = cursor.fetchall()
        
        if activities:
            st.markdown("##### 🔔 Live Feed")
            cols = st.columns(len(activities))
            for idx, row in enumerate(activities):
                with cols[idx]:
                    u_id = row['user_id']
                    # Simple color coding based on user
                    bg_color = "#e3f921" if u_id == 'dad' else "#fce4ec" if u_id == 'mom' else "#e8f5e9"
                    
                    if row['type'] == 'finance':
                        txt = f"Spent €{abs(row['amount']):.0f}" if row['amount'] < 0 else f"Got €{row['amount']:.0f}"
                        icon = "💸"
                    else:
                        txt = f"Exercise {row['exercise_minutes']}m"
                        icon = "🏃"
                        
                    st.markdown(f"""
                    <div style="background-color: {bg_color}; padding: 10px; border-radius: 8px; font-size: 0.8rem;">
                        <strong>{u_id.capitalize()}</strong><br>{icon} {txt}
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown("---")

        # --- Main Tabs ---
        tab1, tab2, tab3 = st.tabs(["📊 Overview", "💰 Family Finance", "🎯 Goals"])

        with tab1:
            # Aggregate Daily Stats
            cursor.execute("SELECT SUM(exercise_minutes), AVG(sleep_hours), AVG(mood_score) FROM health_data WHERE date = date('now')")
            health_today = cursor.fetchone()
            
            cursor.execute("SELECT SUM(ABS(amount)) FROM finance_data WHERE date = date('now') AND type = 'Expense'")
            spend_today = cursor.fetchone()

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                val = health_today[0] if health_today[0] else 0
                st.markdown(f'<div class="family-card"><div class="metric-value">{val}m</div><div class="metric-label">Total Exercise</div></div>', unsafe_allow_html=True)
            with m2:
                val = health_today[1] if health_today[1] else 0
                st.markdown(f'<div class="family-card"><div class="metric-value">{val:.1f}h</div><div class="metric-label">Avg Sleep</div></div>', unsafe_allow_html=True)
            with m3:
                val = health_today[2] if health_today[2] else 0
                st.markdown(f'<div class="family-card"><div class="metric-value">{val:.1f}/10</div><div class="metric-label">Family Mood</div></div>', unsafe_allow_html=True)
            with m4:
                val = spend_today[0] if spend_today[0] else 0
                st.markdown(f'<div class="family-card"><div class="metric-value" style="color:#d63031">€{val:.0f}</div><div class="metric-label">Today\'s Spend</div></div>', unsafe_allow_html=True)

            # Weekly Trends Chart
            st.markdown("### 📅 Weekly Activity")
            df_trend = pd.read_sql_query("SELECT date, user_id, exercise_minutes FROM health_data WHERE date > date('now', '-7 days')", conn)
            if not df_trend.empty:
                fig = px.bar(df_trend, x='date', y='exercise_minutes', color='user_id', barmode='group',
                             color_discrete_map={'dad': '#1976D2', 'mom': '#E91E63', 'teen': '#43A047'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No activity data for the last 7 days.")

        with tab2:
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown("### 💳 Spending Breakdown")
                df_fin = pd.read_sql_query("SELECT category, ABS(amount) as amount FROM finance_data WHERE type='Expense'", conn)
                if not df_fin.empty:
                    fig_pie = px.pie(df_fin, values='amount', names='category', hole=0.4)
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.info("No expense data yet.")
            
            with c2:
                st.markdown("### 📝 Recent Transactions")
                cursor.execute("SELECT date, user_id, amount, category FROM finance_data ORDER BY date DESC LIMIT 5")
                trans = cursor.fetchall()
                for t in trans:
                    color = "red" if t['amount'] < 0 else "green"
                    st.markdown(f"""
                    <div style="border-bottom:1px solid #eee; padding:5px;">
                        <b>{t['user_id'].capitalize()}</b>: <span style="color:{color}">€{abs(t['amount'])}</span> 
                        <br><small>{t['category']} • {t['date']}</small>
                    </div>
                    """, unsafe_allow_html=True)

        with tab3:
            st.markdown("### 🥅 Family Goals")
            
            # Goal Creation
            with st.expander("➕ Create New Goal"):
                with st.form("new_goal"):
                    g_name = st.text_input("Goal Name (e.g., Summer Vacation)")
                    g_target = st.number_input("Target Value", 100.0, 10000.0, 1000.0)
                    g_type = st.selectbox("Type", ["Savings", "Exercise Minutes"])
                    if st.form_submit_button("Set Goal"):
                        conn.execute("INSERT INTO family_goals (goal_type, goal_name, target_value, start_date, end_date, created_by) VALUES (?, ?, ?, date('now'), date('now', '+30 days'), ?)",
                                   (g_type, g_name, g_target, username))
                        conn.commit()
                        st.rerun()

            # Active Goals Logic
            cursor.execute("SELECT * FROM family_goals WHERE status='active'")
            goals = cursor.fetchall()
            
            for goal in goals:
                # Calculate current progress dynamically
                if goal['goal_type'] == "Exercise Minutes":
                    cursor.execute("SELECT SUM(exercise_minutes) FROM health_data WHERE date >= ?", (goal['start_date'],))
                    current = cursor.fetchone()[0] or 0
                    unit = "mins"
                else:
                    # Savings = Income - Expenses
                    cursor.execute("SELECT SUM(amount) FROM finance_data WHERE date >= ?", (goal['start_date'],))
                    current = cursor.fetchone()[0] or 0
                    unit = "€"
                
                # Prevent negative progress visual
                display_current = max(0, current)
                progress = min(display_current / goal['target_value'], 1.0)
                
                st.markdown(f"**{goal['goal_name']}** (Target: {goal['target_value']} {unit})")
                st.progress(progress)
                st.caption(f"Current: {display_current:.0f} {unit} • Started: {goal['start_date']}")
                
                if progress >= 1.0:
                    st.success("🎉 Goal Achieved!")

    elif st.session_state["authentication_status"] is False:
        st.error('Username/password is incorrect')
    elif st.session_state["authentication_status"] is None:
        st.warning('Please enter your username and password')
        st.info("Demo: dad/password123, mom/password123, teen/password123")

if __name__ == "__main__":
    main()