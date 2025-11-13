import matplotlib.pyplot as plt
import numpy as np

# Monthly expenses breakdown
categories = ['Rent', 'Food', 'Transport', 'Entertainment', 'Savings']
amounts = [1200, 750, 200, 400, 650]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

plt.figure(figsize=(10, 6))
bars = plt.bar(categories, amounts, color=colors, alpha=0.8)
plt.title('Monthly Budget Breakdown', fontsize=16, fontweight='bold')
plt.ylabel('Amount (€)')

# Add value labels on bars
for bar, amount in zip(bars, amounts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
             f'€{amount}', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()