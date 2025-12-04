import matplotlib.pyplot as plt
import numpy as np

# Professional styling example
plt.style.use('seaborn-v0_8')  # Modern, clean style

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Left: Default style
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

ax1.plot(x, y1, label='sin(x)', linewidth=2)
ax1.plot(x, y2, label='cos(x)', linewidth=2)
ax1.set_title('Default Matplotlib Style')
ax1.legend()
ax1.grid(True)

# Right: Custom professional style
ax2.plot(x, y1, label='sin(x)', linewidth=3, color='#2E8B57', alpha=0.8)
ax2.plot(x, y2, label='cos(x)', linewidth=3, color='#DC143C', alpha=0.8)
ax2.set_title('Professional Custom Style', fontsize=14, fontweight='bold')
ax2.legend(frameon=True, shadow=True, fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_facecolor('#F8F9FA')

plt.tight_layout()
plt.show()
