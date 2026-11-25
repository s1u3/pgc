import os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs('graphs', exist_ok=True)
os.makedirs('images', exist_ok=True)

# Typography & theme settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Segoe UI', 'Arial']

# Benchmark Data (10,000,000 increments) from Janaki's empirical lab run
threads = [1, 2, 4, 8]
mechanisms = ['None (Race)', 'Atomic', 'Critical', 'Lock (Mutex)']

# Times in seconds directly from lab execution (compare_sync)
times = {
    'None (Race)': [0.0071, 0.0194, 0.0125, 0.0122],
    'Atomic':      [0.0664, 0.1550, 0.1660, 0.1810],
    'Critical':    [0.1143, 0.3579, 0.3421, 0.3618],
    'Lock (Mutex)': [0.1119, 0.3188, 0.3850, 0.3764]
}

# Lost updates percentage for None (Race) from lab output: 10M, 7153581, 3558867, 1856471
final_counts_none = [10000000, 7153581, 3558867, 1856471]
lost_pct = [(1.0 - c / 10000000.0) * 100 for c in final_counts_none]

# Muted, professional engineering palette
colors = {
    'None (Race)': '#64748B',  # Slate Gray
    'Atomic':      '#387864',  # Muted Teal / Sage
    'Critical':    '#B85D43',  # Muted Terracotta
    'Lock (Mutex)': '#2B5C8F'  # Deep Slate Blue
}

# ==============================================================================
# 1. Clear & Readable 2x2 Performance Dashboard
# ==============================================================================
fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), facecolor='#FAFBFC')

# Subplot 1: Execution Time vs Thread Count (Log Scale)
ax1 = axes[0, 0]
for name in mechanisms:
    ax1.plot(threads, times[name], marker='o', linewidth=2.4, markersize=7,
             label=name, color=colors[name])
ax1.set_yscale('log')
ax1.set_title('Execution Time vs. Thread Count (Log Scale)', fontsize=12.5, fontweight='bold', color='#1E293B', pad=10)
ax1.set_xlabel('Thread Count (P)', fontsize=11, color='#334155', fontweight='bold')
ax1.set_ylabel('Execution Time in Seconds (Lower is Better)', fontsize=11, color='#334155', fontweight='bold')
ax1.set_xticks(threads)
ax1.tick_params(axis='both', labelsize=10)
ax1.grid(True, which='both', linestyle=':', color='#E2E8F0', alpha=0.9)
ax1.set_facecolor('#FFFFFF')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.legend(frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1', fontsize=9.5)

# Subplot 2: Grouped Bar Chart of Execution Times
ax2 = axes[0, 1]
x = np.arange(len(threads))
bar_width = 0.2
for i, name in enumerate(mechanisms):
    offset = (i - 1.5) * bar_width
    ax2.bar(x + offset, times[name], bar_width, label=name,
            color=colors[name], edgecolor='#334155', linewidth=1)
ax2.set_yscale('log')
ax2.set_title('Mechanism Comparison Across Thread Counts', fontsize=12.5, fontweight='bold', color='#1E293B', pad=10)
ax2.set_xlabel('Thread Count (P)', fontsize=11, color='#334155', fontweight='bold')
ax2.set_ylabel('Execution Time in Seconds (Log Scale)', fontsize=11, color='#334155', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([f'{t} Threads' for t in threads])
ax2.tick_params(axis='both', labelsize=10)
ax2.grid(True, which='both', linestyle=':', color='#E2E8F0', alpha=0.9)
ax2.set_facecolor('#FFFFFF')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.legend(frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1', fontsize=9.5)

# Subplot 3: Effective Throughput (Million Operations per Second)
ax3 = axes[1, 0]
for name in mechanisms:
    throughput = [10.0 / t for t in times[name]]
    ax3.plot(threads, throughput, marker='s', linewidth=2.4, markersize=7,
             label=name, color=colors[name])
ax3.set_yscale('log')
ax3.set_title('Effective Synchronization Throughput (Mops/s)', fontsize=12.5, fontweight='bold', color='#1E293B', pad=10)
ax3.set_xlabel('Thread Count (P)', fontsize=11, color='#334155', fontweight='bold')
ax3.set_ylabel('Throughput (Mops/sec, Higher is Better)', fontsize=11, color='#334155', fontweight='bold')
ax3.set_xticks(threads)
ax3.tick_params(axis='both', labelsize=10)
ax3.grid(True, which='both', linestyle=':', color='#E2E8F0', alpha=0.9)
ax3.set_facecolor('#FFFFFF')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.legend(frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1', fontsize=9.5)

# Subplot 4: Data Race Counter Loss
ax4 = axes[1, 1]
bars_loss = ax4.bar([f'{t} Threads' for t in threads], lost_pct, color='#B85D43', width=0.48,
                    edgecolor='#334155', linewidth=1)
ax4.set_title('Data Race Integrity Failure: Lost Updates (%)', fontsize=12.5, fontweight='bold', color='#1E293B', pad=10)
ax4.set_xlabel('Thread Count (P)', fontsize=11, color='#334155', fontweight='bold')
ax4.set_ylabel('% of 10M Increments Lost (0% is Ideal)', fontsize=11, color='#334155', fontweight='bold')
ax4.set_ylim(0, 100)
ax4.tick_params(axis='both', labelsize=10)
ax4.grid(True, which='both', linestyle=':', color='#E2E8F0', alpha=0.9)
ax4.set_facecolor('#FFFFFF')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

for bar, pct in zip(bars_loss, lost_pct):
    yval = bar.get_height()
    label = f'{pct:.1f}%' if pct > 0 else '0% (Single Core)'
    ax4.text(bar.get_x() + bar.get_width()/2.0, yval + 2.5, label,
             ha='center', va='bottom', fontsize=10, fontweight='bold', color='#1E293B')

plt.suptitle('Parallel & GPU Computing Lab — Experiment 3: OpenMP Synchronization Benchmark',
             fontsize=14, fontweight='bold', color='#1E293B', y=0.995)
plt.tight_layout()
plt.savefig('graphs/performance_comparison_dashboard.png', dpi=200, bbox_inches='tight')
plt.close()

# ==============================================================================
# 2. Standalone Scalability & Overhead Multiplier Chart
# ==============================================================================
fig, ax = plt.subplots(figsize=(8.5, 5), facecolor='#FAFBFC')

for name in ['Atomic', 'Critical', 'Lock (Mutex)']:
    overhead = [times[name][i] / times[name][0] for i in range(len(threads))]
    ax.plot(threads, overhead, marker='o', linewidth=2.5, markersize=8,
            label=f'{name} (Contention Factor)', color=colors[name])

ax.set_title('Contention Penalty: Execution Time Multiplier Relative to 1 Thread', fontsize=13, fontweight='bold', color='#1E293B', pad=12)
ax.set_xlabel('Thread Count (P)', fontsize=11, color='#334155', fontweight='bold')
ax.set_ylabel('Overhead Multiplier (T_P / T_1, Lower is Better)', fontsize=11, color='#334155', fontweight='bold')
ax.set_xticks(threads)
ax.tick_params(axis='both', labelsize=10)
ax.grid(True, which='both', linestyle=':', color='#E2E8F0', alpha=0.9)
ax.set_facecolor('#FFFFFF')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1', fontsize=10)

plt.tight_layout()
plt.savefig('graphs/contention_overhead_analysis.png', dpi=200, bbox_inches='tight')
plt.close()

print('Experiment 3 charts regenerated with authentic lab data and high readability.')
