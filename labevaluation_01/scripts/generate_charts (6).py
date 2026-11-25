import os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs('images', exist_ok=True)

# Custom typography & theme settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Segoe UI', 'Arial']

threads_num = [1, 2, 4, 8, 12]
threads_labels = ['1 Thread', '2 Threads', '4 Threads\n(Optimal)', '8 Threads', '12 Threads']
times = [0.006786, 0.004173, 0.003273, 0.013438, 0.013632]
speedups = [1.00, 1.63, 2.07, 0.50, 0.50]

# Signature palette: Highlight optimal 4 threads in emerald teal, others distinct
colors = ['#6C5CE7', '#0984E3', '#00B894', '#E17055', '#D63031']
hatches = ['///', '\\\\\\', '***', '...', 'xxx']

# -------------------------------------------------------------
# 1. Combined 2-Panel Benchmark Overview Chart
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2), facecolor='#FAFAFA')

# Panel 1: Execution Time Bar Chart
bars1 = ax1.bar(threads_labels, times, color=colors, width=0.55, edgecolor='#2D3436', linewidth=1.5)
for bar, hatch in zip(bars1, hatches):
    bar.set_hatch(hatch)

ax1.set_title('Execution Time vs. Thread Count (Lower is Better)', fontsize=13, fontweight='bold', color='#2D3436', pad=15)
ax1.set_ylabel('Execution Time in Seconds', fontsize=11, fontweight='bold', color='#636E72')
ax1.grid(True, linestyle=':', color='#DFE6E9', alpha=0.9, axis='y')
ax1.set_facecolor('#FFFFFF')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

for bar, t in zip(bars1, times):
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.0004, f'{t:.6f}s', 
             ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#2D3436',
             bbox=dict(boxstyle='round,pad=0.25', fc='#F1F2F6', ec='#B2BEC3', lw=0.8))

# Panel 2: Speedup Bar Chart with Baseline Line
bars2 = ax2.bar(threads_labels, speedups, color=colors, width=0.55, edgecolor='#2D3436', linewidth=1.5)
for bar, hatch in zip(bars2, hatches):
    bar.set_hatch(hatch)

ax2.axhline(1.0, color='#636E72', linestyle='--', lw=1.8, label='1.0x Baseline (1 Thread)')
ax2.set_title('Speedup Factor relative to 1 Thread (Higher is Better)', fontsize=13, fontweight='bold', color='#2D3436', pad=15)
ax2.set_ylabel('Speedup Ratio (S = T_1 / T_p)', fontsize=11, fontweight='bold', color='#636E72')
ax2.grid(True, linestyle=':', color='#DFE6E9', alpha=0.9, axis='y')
ax2.set_facecolor('#FFFFFF')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.legend(frameon=True, facecolor='#F1F2F6', edgecolor='#B2BEC3', loc='upper right')

for bar, s in zip(bars2, speedups):
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f'{s:.2f}x', 
             ha='center', va='bottom', fontsize=10, fontweight='bold', color='#2D3436',
             bbox=dict(boxstyle='round,pad=0.25', fc='#F1F2F6', ec='#B2BEC3', lw=0.8))

plt.suptitle('Parallel Sum & Average (10 Million Elements) — OpenMP Scalability Analysis', fontsize=15, fontweight='bold', color='#2D3436', y=1.02)
plt.tight_layout()
plt.savefig('images/performance_comparison_charts.png', dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# 2. Standalone Execution Time Chart
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5.2), facecolor='#FAFAFA')
bars = ax.bar(threads_labels, times, color=colors, width=0.52, edgecolor='#2D3436', linewidth=1.5)
for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)

ax.set_title('OpenMP Execution Time for 10M Elements (Lower is Better)', fontsize=13, fontweight='bold', color='#2D3436', pad=15)
ax.set_ylabel('Execution Time (Seconds)', fontsize=11, fontweight='bold', color='#636E72')
ax.grid(True, linestyle=':', color='#DFE6E9', alpha=0.9, axis='y')
ax.set_facecolor('#FFFFFF')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for bar, t in zip(bars, times):
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.0004, f'{t:.6f}s', 
            ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#2D3436',
            bbox=dict(boxstyle='round,pad=0.25', fc='#F1F2F6', ec='#B2BEC3', lw=0.8))

plt.tight_layout()
plt.savefig('images/execution_time_vs_threads.png', dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# 3. Standalone Speedup Chart
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5.2), facecolor='#FAFAFA')
bars = ax.bar(threads_labels, speedups, color=colors, width=0.52, edgecolor='#2D3436', linewidth=1.5)
for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)

ax.axhline(1.0, color='#636E72', linestyle='--', lw=1.8, label='1.0x Baseline (1 Thread)')
ax.set_title('Parallel Speedup Factor vs. Thread Count', fontsize=13, fontweight='bold', color='#2D3436', pad=15)
ax.set_ylabel('Speedup Factor (x)', fontsize=11, fontweight='bold', color='#636E72')
ax.grid(True, linestyle=':', color='#DFE6E9', alpha=0.9, axis='y')
ax.set_facecolor('#FFFFFF')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(frameon=True, facecolor='#F1F2F6', edgecolor='#B2BEC3', loc='upper right')

for bar, s in zip(bars, speedups):
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f'{s:.2f}x', 
            ha='center', va='bottom', fontsize=10, fontweight='bold', color='#2D3436',
            bbox=dict(boxstyle='round,pad=0.25', fc='#F1F2F6', ec='#B2BEC3', lw=0.8))

plt.tight_layout()
plt.savefig('images/speedup_vs_threads.png', dpi=300, bbox_inches='tight')
plt.close()

print('Charts generated successfully in images/')
