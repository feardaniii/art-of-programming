import matplotlib.pyplot as plt
import numpy as np

# How you spend your day
activities = ['Work', 'Sleep', 'Exercise', 'Leisure', 'Commute', 'Meals']
hours = [8, 8, 1, 4, 1.5, 1.5]
colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#FFD700']

plt.figure(figsize=(10, 10))
wedges, texts, autotexts = plt.pie(hours, labels=activities, colors=colors,
                                   autopct='%1.1f%%', startangle=90)

# Make percentage labels bold
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

plt.title('Daily Time Allocation', fontsize=16, fontweight='bold')
plt.axis('equal')
plt.tight_layout()


plt.show()

