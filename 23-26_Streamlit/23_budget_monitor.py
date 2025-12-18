import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict

# Page configuration
st.set_page_config(
    page_title="💰 Smart Budget Monitor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for financial data
if 'expense_data' not in st.session_state:
    st.session_state.expense_data = pd.DataFrame(columns=[
        'date', 'amount', 'category', 'description', 'payment_method'
    ])

if 'budget_limits' not in st.session_state:
    st.session_state.budget_limits = {
        'Food & Dining': 500,
        'Transportation': 200,
        'Shopping': 300,
        'Entertainment': 150,
        'Health': 100,
        'Utilities': 250,
        'Other': 100
    }

if 'savings_goal' not in st.session_state:
    st.session_state.savings_goal = 1000

# Custom CSS for financial styling
st.markdown("""
<style>
.financial-metric {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 0.75rem;
    color: white;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.alert-card {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    padding: 1rem;
    border-radius: 0.5rem;
    color: white;
    margin: 1rem 0;
}
.success-card {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    padding: 1rem;
    border-radius: 0.5rem;
    color: white;
    margin: 1rem 0;
}
.category-card {
    background: white;
    border: 2px solid #e2e8f0;
    border-radius: 0.5rem;
    padding: 1rem;
    transition: all 0.3s ease;
}
.category-card:hover {
    border-color: #667eea;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
}
</style>
""", unsafe_allow_html=True)

# Header
st.title("💰 Smart Budget Monitor")
st.markdown("*Turn financial chaos into intelligent spending decisions*")

# Sidebar for expense input
with st.sidebar:
    st.header("💳 Add New Expense")
    
    # Expense form
    expense_date = st.date_input("Date", datetime.now().date())
    expense_amount = st.number_input("Amount (€)", min_value=0.01, value=10.0, step=0.01)
    
    categories = list(st.session_state.budget_limits.keys())
    expense_category = st.selectbox("Category", categories)
    
    expense_description = st.text_input("Description", placeholder="e.g., Lunch at restaurant")
    
    payment_methods = ["Card", "Cash", "Bank Transfer", "Digital Wallet"]
    payment_method = st.selectbox("Payment Method", payment_methods)
    
    if st.button("💾 Add Expense", type="primary"):
        new_expense = {
            'date': expense_date,
            'amount': expense_amount,
            'category': expense_category,
            'description': expense_description,
            'payment_method': payment_method
        }
        
        st.session_state.expense_data = pd.concat([
            st.session_state.expense_data,
            pd.DataFrame([new_expense])
        ], ignore_index=True)
        
        st.success("✅ Expense added successfully!")
        st.rerun()
    
    st.markdown("---")
    
    # Budget settings
    st.header("🎯 Budget Settings")
    
    # Monthly budget limits
    st.subheader("Monthly Limits")
    for category in categories:
        st.session_state.budget_limits[category] = st.number_input(
            f"{category}",
            min_value=0,
            value=st.session_state.budget_limits[category],
            step=10,
            key=f"budget_{category}"
        )
    
    # Savings goal
    st.session_state.savings_goal = st.number_input(
        "Monthly Savings Goal (€)",
        min_value=0,
        value=st.session_state.savings_goal,
        step=50
    )

# Main dashboard
if len(st.session_state.expense_data) > 0:
    df = st.session_state.expense_data.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Current month filter
    current_month = datetime.now().replace(day=1).date()
    current_month_data = df[df['date'] >= pd.to_datetime(current_month)]
    
    # Key financial metrics
    total_spent = current_month_data['amount'].sum()
    total_budget = sum(st.session_state.budget_limits.values())
    remaining_budget = total_budget - total_spent
    days_left = (datetime.now().replace(day=28).date() - datetime.now().date()).days
    daily_budget_left = remaining_budget / max(days_left, 1) if days_left > 0 else 0
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="financial-metric">
            <h3>💸 Spent This Month</h3>
            <h2>€{total_spent:.2f}</h2>
            <p>of €{total_budget:.2f} budget</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        color = "success-card" if remaining_budget > 0 else "alert-card"
        st.markdown(f"""
        <div class="financial-metric">
            <h3>💰 Remaining Budget</h3>
            <h2>€{remaining_budget:.2f}</h2>
            <p>{"✅ On track!" if remaining_budget > 0 else "⚠️ Over budget!"}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="financial-metric">
            <h3>📅 Daily Budget Left</h3>
            <h2>€{daily_budget_left:.2f}</h2>
            <p>for remaining {days_left} days</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        savings_progress = max(0, remaining_budget)
        savings_percentage = (savings_progress / st.session_state.savings_goal) * 100
        st.markdown(f"""
        <div class="financial-metric">
            <h3>🎯 Savings Progress</h3>
            <h2>{savings_percentage:.1f}%</h2>
            <p>€{savings_progress:.2f} saved</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Budget alerts
    category_spending = current_month_data.groupby('category')['amount'].sum()
    alerts = []
    
    for category, limit in st.session_state.budget_limits.items():
        spent = category_spending.get(category, 0)
        if spent > limit * 0.8:  # Alert at 80% of budget
            percentage = (spent / limit) * 100
            alerts.append(f"⚠️ **{category}**: €{spent:.2f} / €{limit:.2f} ({percentage:.0f}%)")
    
    if alerts:
        st.markdown("""
        <div class="alert-card">
            <h4>🚨 Budget Alerts</h4>
        </div>
        """, unsafe_allow_html=True)
        for alert in alerts:
            st.markdown(alert)
    
    st.markdown("---")
    
    # Interactive analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Trends", "🎯 Categories", "💡 Insights"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly spending trend
            monthly_spending = df.groupby(df['date'].dt.to_period('M'))['amount'].sum()
            fig_monthly = px.line(
                x=monthly_spending.index.astype(str), 
                y=monthly_spending.values,
                title="Monthly Spending Trend",
                labels={'x': 'Month', 'y': 'Amount (€)'}
            )
            fig_monthly.update_traces(line_color='#667eea', line_width=3)
            fig_monthly.update_layout(height=400)
            st.plotly_chart(fig_monthly, use_container_width=True)
        
        with col2:
            # Current month category breakdown
            if not current_month_data.empty:
                category_totals = current_month_data.groupby('category')['amount'].sum()
                fig_pie = px.pie(
                    values=category_totals.values,
                    names=category_totals.index,
                    title="This Month's Spending by Category"
                )
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, use_container_width=True)
    
    with tab2:
        # Daily spending pattern
        daily_spending = df.groupby('date')['amount'].sum().reset_index()
        fig_daily = px.bar(
            daily_spending, 
            x='date', 
            y='amount',
            title="Daily Spending Pattern",
            color='amount',
            color_continuous_scale='Viridis'
        )
        fig_daily.update_layout(height=400)
        st.plotly_chart(fig_daily, use_container_width=True)
        
        # Payment method analysis
        payment_analysis = df.groupby('payment_method')['amount'].sum()
        fig_payment = px.bar(
            x=payment_analysis.index,
            y=payment_analysis.values,
            title="Spending by Payment Method",
            labels={'x': 'Payment Method', 'y': 'Total Amount (€)'}
        )
        fig_payment.update_layout(height=300)
        st.plotly_chart(fig_payment, use_container_width=True)
    
    with tab3:
        # Budget vs actual by category
        budget_comparison = []
        for category in categories:
            spent = category_spending.get(category, 0)
            budget = st.session_state.budget_limits[category]
            budget_comparison.append({
                'Category': category,
                'Spent': spent,
                'Budget': budget,
                'Remaining': budget - spent,
                'Usage %': (spent / budget) * 100 if budget > 0 else 0
            })
        
        budget_df = pd.DataFrame(budget_comparison)
        
        # Visual budget comparison
        fig_budget = go.Figure()
        fig_budget.add_trace(go.Bar(
            name='Spent',
            x=budget_df['Category'],
            y=budget_df['Spent'],
            marker_color='#f5576c'
        ))
        fig_budget.add_trace(go.Bar(
            name='Remaining Budget',
            x=budget_df['Category'],
            y=budget_df['Remaining'],
            marker_color='#4facfe'
        ))
        
        fig_budget.update_layout(
            title="Budget vs Actual Spending by Category",
            barmode='stack',
            height=400
        )
        st.plotly_chart(fig_budget, use_container_width=True)
        
        # Category details table
        st.dataframe(
            budget_df[['Category', 'Spent', 'Budget', 'Usage %']].round(2),
            use_container_width=True
        )
    
    with tab4:
        # Financial insights
        st.markdown("""
        <div class="success-card">
            <h3>🧠 AI-Powered Financial Insights</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if len(df) >= 7:  # Need at least a week of data
            # Most expensive category
            top_category = category_spending.idxmax()
            top_amount = category_spending.max()
            
            st.success(f"💡 **Biggest spending category**: {top_category} (€{top_amount:.2f})")
            
            # Spending velocity
            recent_week = df[df['date'] >= (datetime.now() - timedelta(days=7))]
            recent_spending = recent_week['amount'].sum()
            weekly_average = df['amount'].sum() / ((df['date'].max() - df['date'].min()).days / 7)
            
            if recent_spending > weekly_average * 1.2:
                st.warning(f"⚠️ **Spending acceleration detected!** This week: €{recent_spending:.2f} (avg: €{weekly_average:.2f})")
            else:
                st.success(f"✅ **Spending on track!** This week: €{recent_spending:.2f}")
            
            # Day of week analysis
            df['day_of_week'] = df['date'].dt.day_name()
            day_spending = df.groupby('day_of_week')['amount'].mean()
            highest_day = day_spending.idxmax()
            
            st.info(f"📅 **Highest spending day**: {highest_day} (avg: €{day_spending[highest_day]:.2f})")
            
            # Savings projection
            if remaining_budget > 0:
                annual_savings = remaining_budget * 12
                st.success(f"🎯 **Annual savings projection**: €{annual_savings:.2f} if you maintain this pace!")
        
        else:
            st.info("📊 Add more expense data to unlock AI-powered insights!")
    
    # Data export and management
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Export Data as CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download expense_data.csv",
                data=csv,
                file_name=f"expenses_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("🗑️ Clear All Data"):
            st.session_state.expense_data = pd.DataFrame(columns=[
                'date', 'amount', 'category', 'description', 'payment_method'
            ])
            st.rerun()
    
    with col3:
        if st.button("📊 Monthly Report"):
            st.balloons()
            st.success("Monthly report generated! Check the insights tab for detailed analysis.")

else:
    st.info("👆 Start by adding your first expense in the sidebar!")
    
    # Demo data button
    if st.button("💡 Load Demo Data (2 weeks)"):
        demo_categories = list(st.session_state.budget_limits.keys())
        demo_data = []
        
        for i in range(14):
            date = datetime.now().date() - timedelta(days=i)
            # Generate 1-4 expenses per day
            for _ in range(np.random.randint(1, 5)):
                demo_data.append({
                    'date': date,
                    'amount': round(np.random.uniform(5, 150), 2),
                    'category': np.random.choice(demo_categories),
                    'description': f"Sample expense {len(demo_data) + 1}",
                    'payment_method': np.random.choice(["Card", "Cash", "Bank Transfer"])
                })
        
        st.session_state.expense_data = pd.DataFrame(demo_data)
        st.success("Demo financial data loaded! Refresh to see your budget dashboard.")
        st.rerun()

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit • Master your money, master your life 💰*")