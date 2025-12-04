
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Financial dashboard layout
fig = plt.figure(figsize=(16, 12))

# Define a complex grid
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Main chart: Portfolio performance
ax1 = fig.add_subplot(gs[0, :])


# Date Folosite
portfolio_dates = pd.date_range('2024-01-01', periods=50, freq='W')
portfolio_value = 10000 + np.cumsum(np.random.randn(50) * 100)



ax1.plot(portfolio_dates, portfolio_value, linewidth=3, color='#2E8B57')
ax1.set_title('Portfolio Performance', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Asset allocation pie chart
ax2 = fig.add_subplot(gs[1, 0])
assets = ['Stocks', 'Bonds', 'Real Estate', 'Cash']
allocation = [60, 25, 10, 5]
ax2.pie(allocation, labels=assets, autopct='%1.1f%%')
ax2.set_title('Asset Allocation')

# Monthly returns bar chart
ax3 = fig.add_subplot(gs[1, 1])
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
returns = [2.5, -1.2, 3.8, 1.9, -0.5, 2.1]
bars = ax3.bar(months, returns, color=['green' if x > 0 else 'red' for x in returns])
ax3.set_title('Monthly Returns (%)')
ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)

# Risk metrics
ax4 = fig.add_subplot(gs[1, 2])
metrics = ['Volatility', 'Sharpe Ratio', 'Max Drawdown']
values = [0.15, 1.2, -0.08]
ax4.barh(metrics, values, color='#FFD700')
ax4.set_title('Risk Metrics')

# Bottom row: correlation matrix heatmap simulation
ax5 = fig.add_subplot(gs[2, :])
corr_data = np.random.rand(5, 5)
corr_data = (corr_data + corr_data.T) / 2  # Make symmetric
np.fill_diagonal(corr_data, 1)  # Diagonal should be 1
im = ax5.imshow(corr_data, cmap='coolwarm', aspect='auto')
ax5.set_title('Asset Correlation Matrix')
ax5.set_xticks(range(5))
ax5.set_yticks(range(5))
ax5.set_xticklabels(['Stock A', 'Stock B', 'Bond A', 'REIT', 'Gold'])
ax5.set_yticklabels(['Stock A', 'Stock B', 'Bond A', 'REIT', 'Gold'])

plt.tight_layout()
plt.show()