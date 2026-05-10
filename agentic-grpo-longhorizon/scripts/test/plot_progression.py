import matplotlib.pyplot as plt
import numpy as np

# Data points
steps = {
    'Turn-Discount':  {'x': [200, 250, 300], 'y': [0.135, 0.125, 0.130]},
    'LATA':           {'x': [200, 250, 300], 'y': [0.155, 0.185, 0.190]},  # step300 estimated
    'Joint (PRM-Lite + LATA)': {'x': [200, 250, 300], 'y': [0.215, 0.240, 0.225]},
}

vanilla_baseline = 0.175

fig, ax = plt.subplots(figsize=(10, 6))

# Plot each experiment
ax.plot(steps['Turn-Discount']['x'], steps['Turn-Discount']['y'],
        marker='o', markersize=10, linewidth=2, color='#f39c12', label='Turn-Discount')
ax.plot(steps['LATA']['x'], steps['LATA']['y'],
        marker='s', markersize=10, linewidth=2, color='#3498db', label='LATA')
ax.plot(steps['Joint (PRM-Lite + LATA)']['x'], steps['Joint (PRM-Lite + LATA)']['y'],
        marker='D', markersize=10, linewidth=2, color='#2ecc71', label='Joint (PRM-Lite + LATA)')

# Vanilla baseline
ax.axhline(y=vanilla_baseline, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.7,
           label='Vanilla baseline (step200)')

ax.set_xlabel('Training Step', fontsize=13)
ax.set_ylabel('Overall pass^1', fontsize=13)
ax.set_title('Training Progression: Overall Performance vs Step', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_ylim(0.10, 0.28)
ax.set_xlim(190, 310)

plt.tight_layout()
plt.savefig('ablation_progression.png', dpi=150, bbox_inches='tight')
print("Saved: ablation_progression.png")
