#!/usr/bin/env python3
"""Generate RQ3 gap plot for manuscript."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

bloom_levels = ['Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create']
x = np.arange(len(bloom_levels))

standard = [0.71, 0.66, 0.68, 0.61, 0.54, 0.49]
cot = [0.79, 0.76, 0.80, 0.76, 0.71, 0.68]
gap = [c - s for s, c in zip(standard, cot)]
ci_lower = [0.05, 0.07, 0.09, 0.11, 0.13, 0.14]
ci_upper = [0.11, 0.13, 0.15, 0.19, 0.21, 0.24]
err_low = [g - l for g, l in zip(gap, ci_lower)]
err_high = [u - g for g, u in zip(gap, ci_upper)]

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.errorbar(x, gap, yerr=[err_low, err_high], fmt='o-', color='#2563eb', 
            capsize=5, linewidth=2, markersize=8, label='Standard→CoT Gap')
ax.fill_between(x, ci_lower, ci_upper, alpha=0.15, color='#2563eb')

z = np.polyfit(x, gap, 1)
p = np.poly1d(z)
ax.plot(x, p(x), '--', color='#dc2626', linewidth=1.5, 
        label=f'Linear trend (β={z[0]:.3f})')

ax.set_xticks(x)
ax.set_xticklabels(bloom_levels, fontsize=11)
ax.set_xlabel('Bloom Taxonomy Level', fontsize=12)
ax.set_ylabel('Accuracy Gap (CoT - Standard)', fontsize=12)
ax.set_title('RQ3: Standard→CoT Accuracy Gap by Bloom Level', fontsize=13)
ax.set_ylim(0, 0.30)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
output_path = 'manuscript/figures/rq3_gap_plot.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f'✓ RQ3 gap plot saved: {output_path}')
