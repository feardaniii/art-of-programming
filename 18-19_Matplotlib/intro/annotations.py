import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Stock price with key events annotated
dates = pd.date_range('2024-01-01', periods=100, freq='D')
price = 100 + np.cumsum(np.random.randn(100) * 0.5)

plt.figure(figsize=(12, 8))
plt.plot(dates, price, linewidth=2, color='#2E8B57')

# Highlight key events
plt.annotate('Product Launch',
             xy=(dates[30], price[30]), xytext=(dates[40], price[30] + 10),
             arrowprops=dict(arrowstyle='->', color='red', lw=2),
             fontsize=12, fontweight='bold', color='red')

plt.annotate('Market Crash',
             xy=(dates[70], price[70]), xytext=(dates[60], price[70] - 10),
             arrowprops=dict(arrowstyle='->', color='blue', lw=2),
             fontsize=12, fontweight='bold', color='blue')

plt.title('Stock Price with Key Events', fontsize=16, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()