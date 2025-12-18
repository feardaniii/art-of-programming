import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="💰 AI Finance Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Session State ---
if 'finance_data' not in st.session_state:
    st.session_state.finance_data = pd.DataFrame()
if 'anomaly_model' not in st.session_state:
    st.session_state.anomaly_model = None
if 'spending_forecast' not in st.session_state:
    st.session_state.spending_forecast = None

# --- Styling ---
st.markdown("""
<style>
.ai-insight {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    padding: 1.5rem;
    border-radius: 0.75rem;
    color: white;
    margin: 1rem 0;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.anomaly-alert {
    background: linear-gradient(135deg, #ff6b6b 0%, #ee5253 100%);
    padding: 1rem;
    border-radius: 0.5rem;
    color: white;
    margin: 0.5rem 0;
    border-left: 5px solid #c0392b;
}
.forecast-card {
    background: white;
    padding: 1.5rem;
    border-radius: 0.75rem;
    color: #2d3436;
    text-align: center;
    border: 1px solid #dfe6e9;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
.metric-label {
    font-size: 0.9rem;
    color: #b2bec3;
    text-transform: uppercase;
    letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def prepare_features(df, features_to_keep=None):
    """
    Robust feature engineering that guarantees consistent columns
    between Training and Prediction phases.
    """
    # Create dummies
    df_encoded = pd.get_dummies(df, columns=['category', 'payment_method'])
    
    # Base numeric features
    base_features = ['amount', 'day_of_week', 'day_of_month', 'is_weekend', 'month']
    
    # Identify all available columns
    all_features = base_features + [col for col in df_encoded.columns if col.startswith(('category_', 'payment_method_'))]
    
    # If we are PREDICTING (features_to_keep exists), enforce the structure
    if features_to_keep:
        # 1. Keep only columns that exist in both, or add missing ones as 0
        df_ready = df_encoded.reindex(columns=features_to_keep, fill_value=0)
        return df_ready[features_to_keep]
    
    # If we are TRAINING, return everything and the list of columns used
    return df_encoded[all_features], all_features

# --- Sidebar ---
with st.sidebar:
    st.title("💳 Wallet AI")
    
    st.header("New Transaction")
    
    with st.expander("📝 Details", expanded=True):
        trans_date = st.date_input("Date", datetime.now().date())
        amount = st.number_input("Amount (€)", min_value=0.01, value=50.0, step=10.0)
        transaction_type = st.radio("Type", ["Expense", "Income"], horizontal=True)
        
        category = st.selectbox("Category", [
            "Food & Dining", "Transportation", "Shopping", "Entertainment",
            "Bills & Utilities", "Healthcare", "Travel", "Education",
            "Groceries", "Gas", "Salary", "Investment", "Other"
        ])
    
    with st.expander("⚙️ Context", expanded=False):
        description = st.text_input("Description", "Regular purchase")
        merchant = st.text_input("Merchant", "")
        payment_method = st.selectbox("Method", ["Card", "Cash", "Transfer", "Digital"])
    
    if st.button("💾 Add Transaction", type="primary"):
        # Store expenses as POSITIVE for logic, but mark type clearly
        final_amount = amount if transaction_type == "Expense" else amount 
        
        new_transaction = {
            'date': trans_date,
            'amount': final_amount, 
            'real_value': -amount if transaction_type == "Expense" else amount, # For net calc
            'category': category,
            'type': transaction_type,
            'description': description,
            'merchant': merchant,
            'payment_method': payment_method,
            'day_of_week': trans_date.weekday(),
            'day_of_month': trans_date.day,
            'is_weekend': int(trans_date.weekday() >= 5),
            'month': trans_date.month
        }
        
        st.session_state.finance_data = pd.concat([
            st.session_state.finance_data,
            pd.DataFrame([new_transaction])
        ], ignore_index=True)
        
        st.toast("Transaction Added!", icon="✅")
    
    st.markdown("---")
    
    # --- AI Controls ---
    st.header("🤖 AI Brain")
    
    df = st.session_state.finance_data
    
    # Only show controls if we have data
    if len(df) > 5:
        # 1. Anomaly Detector
        st.subheader("1. Anomaly Scanner")
        contamination = st.slider("Sensitivity", 0.01, 0.20, 0.05, help="Higher = Find more anomalies")
        
        if st.button("🔍 Scan for Anomalies"):
            # Filter for Expenses only (anomalies in income are usually good!)
            expense_data = df[df['type'] == 'Expense'].copy()
            
            if len(expense_data) < 5:
                st.error("Need at least 5 expense transactions.")
            else:
                # Prepare features
                X, feature_names = prepare_features(expense_data)
                X = X.fillna(0)
                
                # Scale
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Train
                iso_forest = IsolationForest(contamination=contamination, random_state=42)
                iso_forest.fit(X_scaled)
                
                # Save
                st.session_state.anomaly_model = {
                    'model': iso_forest,
                    'scaler': scaler,
                    'features': feature_names,
                    'last_run': datetime.now()
                }
                st.success("Model Trained!")
                st.rerun()

        # 2. Forecasting
        st.subheader("2. Future Cast")
        if st.button("📈 Forecast Cash Flow"):
            # Aggregate by month
            df_time = df.copy()
            df_time['date'] = pd.to_datetime(df_time['date'])
            # Resample to ensure all months are present (fills gaps with 0)
            monthly = df_time.set_index('date').resample('M')['real_value'].sum().reset_index()
            
            # Create simple time features for Linear Regression
            monthly['month_idx'] = range(len(monthly))
            
            if len(monthly) >= 2:
                X = monthly[['month_idx']]
                y = monthly['real_value']
                
                model = LinearRegression()
                model.fit(X, y)
                
                # Predict next 3 months
                last_idx = monthly['month_idx'].max()
                future_idx = pd.DataFrame({'month_idx': [last_idx+1, last_idx+2, last_idx+3]})
                predictions = model.predict(future_idx)
                
                # Calculate simple confidence (based on std dev of errors)
                residuals = y - model.predict(X)
                std_dev = residuals.std()
                
                future_dates = [monthly['date'].iloc[-1] + pd.DateOffset(months=i) for i in range(1, 4)]
                
                st.session_state.spending_forecast = {
                    'dates': future_dates,
                    'values': predictions,
                    'std_dev': std_dev
                }
                st.success("Forecast Generated!")
                st.rerun()
            else:
                st.warning("Need data spanning at least 2 months.")

# --- Main Dashboard ---
st.title("💰 AI-Powered Finance Dashboard")

if len(st.session_state.finance_data) > 0:
    df = st.session_state.finance_data.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Top Metrics
    current_month = datetime.now().month
    mask_curr = df['date'].dt.month == current_month
    
    # Calculate flows
    expenses = df[mask_curr & (df['type']=='Expense')]['amount'].sum()
    income = df[mask_curr & (df['type']=='Income')]['amount'].sum()
    balance = income - expenses
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Expenses (This Month)", f"€{expenses:,.2f}", delta=None)
    with col2:
        st.metric("Income (This Month)", f"€{income:,.2f}", delta=None)
    with col3:
        st.metric("Net Flow", f"€{balance:,.2f}", delta_color="normal")
    with col4:
        # AI Quick Stat
        if st.session_state.spending_forecast:
            next_m = st.session_state.spending_forecast['values'][0]
            st.metric("🤖 Next Month Predicted", f"€{next_m:,.2f}", 
                     delta=f"{next_m - balance:,.2f} vs now", delta_color="inverse")
        else:
            st.info("Train AI for prediction")

    st.markdown("---")

    # --- AI Analysis Section ---
    col_ai_1, col_ai_2 = st.columns([1, 1])
    
    with col_ai_1:
        st.subheader("🕵️ Anomaly Detection")
        if st.session_state.anomaly_model:
            model_pkg = st.session_state.anomaly_model
            
            # Run prediction on ALL expenses to visualize
            expense_data = df[df['type'] == 'Expense'].copy()
            
            if not expense_data.empty:
                # Use the ROBUST prepare function
                X_pred = prepare_features(expense_data, features_to_keep=model_pkg['features'])
                X_scaled = model_pkg['scaler'].transform(X_pred)
                
                # Predict (-1 is anomaly, 1 is normal)
                expense_data['anomaly_score'] = model_pkg['model'].predict(X_scaled)
                anomalies = expense_data[expense_data['anomaly_score'] == -1]
                
                if not anomalies.empty:
                    st.error(f"⚠️ Found {len(anomalies)} suspicious transactions!")
                    
                    # Visualization
                    fig_anom = px.scatter(
                        expense_data, x='date', y='amount', 
                        color=expense_data['anomaly_score'].map({1: 'Normal', -1: 'Anomaly'}),
                        color_discrete_map={'Normal': '#00b894', 'Anomaly': '#d63031'},
                        hover_data=['category', 'description'],
                        title="Transaction Normality Distribution"
                    )
                    st.plotly_chart(fig_anom, use_container_width=True)
                    
                    # List details
                    with st.expander("🔍 View Suspicious Details", expanded=True):
                        for _, row in anomalies.iterrows():
                            st.markdown(f"""
                            <div class="anomaly-alert">
                                <b>{row['date'].strftime('%Y-%m-%d')}</b>: €{row['amount']} | {row['category']}<br>
                                <small>"{row['description']}" - Unusual pattern detected</small>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.success("✅ All transactions look normal based on your history.")
        else:
            st.info("👈 Click 'Scan for Anomalies' in the sidebar to analyze spending patterns.")

    with col_ai_2:
        st.subheader("🔮 Cash Flow Forecast")
        if st.session_state.spending_forecast:
            forecast = st.session_state.spending_forecast
            
            # Prepare data for plotting
            dates = forecast['dates']
            values = forecast['values']
            std = forecast['std_dev']
            
            # Visual Cards
            f_col1, f_col2, f_col3 = st.columns(3)
            for i, (col, d, v) in enumerate(zip([f_col1, f_col2, f_col3], dates, values)):
                color = "green" if v > 0 else "red"
                with col:
                    st.markdown(f"""
                    <div class="forecast-card">
                        <div class="metric-label">{d.strftime('%b %Y')}</div>
                        <h3 style="color: {color}">€{v:,.0f}</h3>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Trend Chart with Confidence Interval
            hist_df = df.copy()
            hist_df['date'] = pd.to_datetime(hist_df['date'])
            monthly_hist = hist_df.set_index('date').resample('M')['real_value'].sum().reset_index()
            
            fig_cast = go.Figure()
            
            # History Line
            fig_cast.add_trace(go.Scatter(
                x=monthly_hist['date'], y=monthly_hist['real_value'],
                name='History', mode='lines+markers', line=dict(color='#636e72')
            ))
            
            # Forecast Line
            fig_cast.add_trace(go.Scatter(
                x=dates, y=values,
                name='Forecast', mode='lines+markers', line=dict(color='#0984e3', width=3, dash='dot')
            ))
            
            # Confidence Interval (Upper/Lower bounds)
            fig_cast.add_trace(go.Scatter(
                x=dates + dates[::-1], # X coordinates for polygon
                y=list(values + std) + list(values - std)[::-1], # Upper then Lower reversed
                fill='toself',
                fillcolor='rgba(9, 132, 227, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Range'
            ))
            
            fig_cast.update_layout(title="Cash Flow Trajectory", height=350)
            st.plotly_chart(fig_cast, use_container_width=True)
            
        else:
            st.info("👈 Click 'Forecast Cash Flow' in the sidebar to see the future.")

else:
    st.info("👋 Start by adding a transaction or load the demo data below.")
    
    if st.button("🎲 Load Smart Demo Data"):
        # Generate correlated data for better AI demonstration
        dates = [datetime.now() - timedelta(days=x) for x in range(120)]
        data = []
        
        for d in dates:
            # Monthly Salary
            if d.day == 25:
                data.append(['Income', 2500, 'Salary', 'Salary', 'Transfer', d])
            
            # Weekly Groceries
            if d.weekday() == 5: # Saturday
                amt = np.random.normal(120, 20)
                data.append(['Expense', amt, 'Groceries', 'Weekly Shop', 'Card', d])
            
            # Daily small stuff
            if np.random.random() > 0.3:
                cat = np.random.choice(['Food', 'Transport', 'Coffee'])
                amt = np.random.normal(15, 5)
                data.append(['Expense', amt, cat, 'Daily spend', 'Card', d])
                
            # Random Anomaly (High value expense)
            if np.random.random() > 0.98:
                data.append(['Expense', 800, 'Shopping', 'Impulse Buy', 'Card', d])

        # Create DataFrame
        demo_df = pd.DataFrame(data, columns=['type', 'amount', 'category', 'description', 'payment_method', 'date'])
        
        # Add derived columns
        demo_df['real_value'] = demo_df.apply(lambda x: x['amount'] if x['type'] == 'Income' else -x['amount'], axis=1)
        demo_df['day_of_week'] = demo_df['date'].dt.weekday
        demo_df['day_of_month'] = demo_df['date'].dt.day
        demo_df['is_weekend'] = (demo_df['day_of_week'] >= 5).astype(int)
        demo_df['month'] = demo_df['date'].dt.month
        demo_df['merchant'] = "Demo Merchant"
        
        st.session_state.finance_data = demo_df
        st.success("✅ Smart data loaded! Try the AI buttons now.")
        st.rerun()