import matplotlib.pyplot as plt
import numpy as np

# Your daily step count over 2 weeks
days = range(1, 15)
steps = [8500, 7200, 9100, 10200, 6800, 9500, 11200, 
         8800, 7600, 9800, 10500, 8900, 7400, 9200]

plt.figure(figsize=(10, 6))
plt.plot(days, steps, marker='o', linewidth=2, color='#2E8B57')
plt.title('Your Daily Steps: 2-Week Progress', fontsize=16, fontweight='bold')
plt.xlabel('Day')
plt.ylabel('Steps')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()