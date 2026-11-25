#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs('images', exist_ok=True)

# Typography & theme settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Segoe UI', 'Arial']

# Muted, professional engineering palette
c_blue = '#2B5C8F'      # Deep Slate Blue
c_red = '#B85D43'       # Muted Terracotta (Adjacent / False Sharing)
c_green = '#387864'     # Muted Teal / Sage (Padded / Aligned)
c_slate = '#64748B'     # Slate
c_amber = '#C07D3E'     # Warm Amber
c_dark = '#1E293B'      # Dark Charcoal
c_gray = '#334155'      # Medium Slate

# =========================================================================
# 1. Cache Coherence Protocol Bus Transactions & Memory Traffic Chart
# =========================================================================
protocols = ['MSI', 'MESI', 'MOESI']
adjacent_bus_tx = [4000, 3999, 3999]
padded_bus_tx = [4, 2, 2]
writebacks = [1999, 1999, 0]
c2c_transfers = [0, 0, 1999]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), facecolor='#FAFBFC')

# Panel 1: Bus Transactions
x = np.arange(len(protocols))
width = 0.35

rects1 = ax1.bar(x - width/2, adjacent_bus_tx, width, label='Adjacent (False Sharing)', color=c_red, edgecolor='#334155', linewidth=1)
rects2 = ax1.bar(x + width/2, padded_bus_tx, width, label='Padded (Cache Aligned)', color=c_green, edgecolor='#334155', linewidth=1)

ax1.set_title('Bus Transactions: False Sharing vs Padded', fontsize=12.5, fontweight='bold', color=c_dark, pad=12)
ax1.set_xlabel('Coherence Protocol', fontsize=11, color=c_gray, fontweight='bold')
ax1.set_ylabel('Total Bus Transactions (Lower is Better)', fontsize=11, color=c_gray, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(protocols, fontsize=10.5, fontweight='bold')
ax1.tick_params(axis='y', labelsize=10)
ax1.set_facecolor('#FFFFFF')
ax1.grid(True, linestyle=':', color='#E2E8F0', alpha=0.9)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.legend(frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1', fontsize=9.5)

for r in rects1:
    h = r.get_height()
    ax1.annotate(f'{h}', xy=(r.get_x() + r.get_width()/2, h),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=c_dark)

for r in rects2:
    h = r.get_height()
    ax1.annotate(f'{h}', xy=(r.get_x() + r.get_width()/2, h),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=c_dark)

# Panel 2: Memory Traffic - Writeback vs Cache-to-Cache Transfer
rects3 = ax2.bar(x - width/2, writebacks, width, label='Forced DRAM Writebacks', color=c_amber, edgecolor='#334155', linewidth=1)
rects4 = ax2.bar(x + width/2, c2c_transfers, width, label='Cache-to-Cache Transfers (Owned)', color=c_blue, edgecolor='#334155', linewidth=1)

ax2.set_title('DRAM Writebacks vs Cache-to-Cache Servicing', fontsize=12.5, fontweight='bold', color=c_dark, pad=12)
ax2.set_xlabel('Coherence Protocol', fontsize=11, color=c_gray, fontweight='bold')
ax2.set_ylabel('Transaction Count', fontsize=11, color=c_gray, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(protocols, fontsize=10.5, fontweight='bold')
ax2.tick_params(axis='y', labelsize=10)
ax2.set_facecolor('#FFFFFF')
ax2.grid(True, linestyle=':', color='#E2E8F0', alpha=0.9)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.legend(frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1', fontsize=9.5)

for r in rects3:
    h = r.get_height()
    ax2.annotate(f'{h}', xy=(r.get_x() + r.get_width()/2, h),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=c_dark)

for r in rects4:
    h = r.get_height()
    ax2.annotate(f'{h}', xy=(r.get_x() + r.get_width()/2, h),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=c_dark)

plt.tight_layout()
plt.savefig('images/cache_coherence_bus_traffic.png', dpi=200, bbox_inches='tight')
plt.close()

# =========================================================================
# 2. Empirical False Sharing Execution Time & Speedup Scaling Chart
# =========================================================================
threads = [1, 2, 4]
adj_time = [0.0207, 0.0176, 0.0216]
pad_time = [0.0204, 0.0121, 0.0118]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), facecolor='#FAFBFC')

# Panel 1: Execution Time
ax1.plot(threads, adj_time, marker='o', markersize=8, color=c_red, lw=2.5, label='Adjacent (False Sharing Bottleneck)')
ax1.plot(threads, pad_time, marker='s', markersize=8, color=c_green, lw=2.5, linestyle='--', label='Padded (64B Aligned Scaling)')

for t, at, pt in zip(threads, adj_time, pad_time):
    ax1.annotate(f'{at:.4f}s', (t, at), xytext=(-10, 7), textcoords='offset points', fontsize=9.5, fontweight='bold', color=c_dark)
    ax1.annotate(f'{pt:.4f}s', (t, pt), xytext=(-10, -14), textcoords='offset points', fontsize=9.5, fontweight='bold', color=c_dark)

ax1.set_title('Empirical Execution Time: 100M Increments', fontsize=12.5, fontweight='bold', color=c_dark, pad=12)
ax1.set_xlabel('Thread Count (P)', fontsize=11, color=c_gray, fontweight='bold')
ax1.set_ylabel('Execution Time in Seconds (Lower is Better)', fontsize=11, color=c_gray, fontweight='bold')
ax1.set_xticks(threads)
ax1.tick_params(axis='both', labelsize=10)
ax1.set_ylim(0.008, 0.026)
ax1.set_facecolor('#FFFFFF')
ax1.grid(True, linestyle=':', color='#E2E8F0', alpha=0.9)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.legend(frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1', fontsize=9.5)

# Panel 2: Speedup Advantage of Padded Layout
speedup_advantage = [at / pt for at, pt in zip(adj_time, pad_time)]
x_pos = np.arange(len(threads))

bars = ax2.bar(x_pos, speedup_advantage, width=0.48, color=[c_amber, c_slate, c_green], edgecolor='#334155', linewidth=1)
for b, val in zip(bars, speedup_advantage):
    ax2.annotate(f'{val:.2f}x\nFaster', xy=(b.get_x() + b.get_width()/2, val),
                 xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=c_dark)

ax2.set_title('Padded Layout Speedup Gain over Adjacent', fontsize=12.5, fontweight='bold', color=c_dark, pad=12)
ax2.set_xlabel('Thread Count (P)', fontsize=11, color=c_gray, fontweight='bold')
ax2.set_ylabel('Speedup Factor vs. Adjacent (x)', fontsize=11, color=c_gray, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f'{t} Thread{"s" if t > 1 else ""}' for t in threads], fontsize=10.5, fontweight='bold')
ax2.tick_params(axis='y', labelsize=10)
ax2.set_ylim(0, 2.5)
ax2.set_facecolor('#FFFFFF')
ax2.grid(True, linestyle=':', color='#E2E8F0', alpha=0.9)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('images/false_sharing_scaling.png', dpi=200, bbox_inches='tight')
plt.close()

# =========================================================================
# 3. NUMA Memory Sweep Throughput
# =========================================================================
labels = ['1T Default', '1T Node 0 Bound', '2T Default', '2T Node 0 Bound']
sweep_times = [0.1152, 0.1027, 0.1108, 0.0984]
total_gb = (512 * 4) / 1024.0  # 2.0 GB
sweep_bw = [total_gb / t for t in sweep_times]

fig, ax1 = plt.subplots(figsize=(8.5, 5), facecolor='#FAFBFC')
bar_colors = [c_blue, c_slate, c_blue, c_green]
bars = ax1.bar(labels, sweep_bw, color=bar_colors, width=0.5, edgecolor='#334155', linewidth=1)

for b, t_val, bw_val in zip(bars, sweep_times, sweep_bw):
    ax1.annotate(f'{bw_val:.2f} GB/s\n({t_val:.4f}s)', xy=(b.get_x() + b.get_width()/2, bw_val),
                 xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=c_dark)

ax1.set_title('NUMA 512 MB Memory Sweep: Throughput & Latency', fontsize=13, fontweight='bold', color=c_dark, pad=12)
ax1.set_ylabel('Effective Memory Bandwidth (GB/s)', fontsize=11, color=c_gray, fontweight='bold')
ax1.set_ylim(0, 26)
ax1.tick_params(axis='both', labelsize=10)
ax1.grid(True, linestyle=':', color='#E2E8F0', alpha=0.9)
ax1.set_facecolor('#FFFFFF')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('images/numa_sweep_latency.png', dpi=200, bbox_inches='tight')
plt.close()

# =========================================================================
# 4. Master 4-Panel Performance Dashboard
# =========================================================================
fig, ((d1, d2), (d3, d4)) = plt.subplots(2, 2, figsize=(13.5, 9.5), facecolor='#FAFBFC')

# Panel 1: Protocol Bus Transactions
d1.bar(x - width/2, adjacent_bus_tx, width, label='Adjacent (False Sharing)', color=c_red, edgecolor='#334155', linewidth=1)
d1.bar(x + width/2, padded_bus_tx, width, label='Padded (Cache Aligned)', color=c_green, edgecolor='#334155', linewidth=1)
d1.set_title('(A) Cache Coherence Bus Transactions', fontsize=12, fontweight='bold', color=c_dark)
d1.set_xticks(x)
d1.set_xticklabels(protocols, fontsize=10, fontweight='bold')
d1.set_ylabel('Total Bus Transactions', fontsize=10.5, color=c_gray, fontweight='bold')
d1.tick_params(axis='both', labelsize=9.5)
d1.grid(True, linestyle=':', color='#E2E8F0', alpha=0.9)
d1.set_facecolor('#FFFFFF')
d1.legend(frameon=True, facecolor='#F8FAFC', fontsize=9)

# Panel 2: False Sharing Degradation vs Padded Scaling
d2.plot(threads, adj_time, marker='o', color=c_red, lw=2.4, label='Adjacent (False Sharing)')
d2.plot(threads, pad_time, marker='s', color=c_green, lw=2.4, linestyle='--', label='Padded (Aligned)')
d2.set_title('(B) False Sharing Penalty: 100M Increments', fontsize=12, fontweight='bold', color=c_dark)
d2.set_xticks(threads)
d2.set_xlabel('Thread Count (P)', fontsize=10.5, color=c_gray, fontweight='bold')
d2.set_ylabel('Execution Time in Seconds', fontsize=10.5, color=c_gray, fontweight='bold')
d2.tick_params(axis='both', labelsize=9.5)
d2.grid(True, linestyle=':', color='#E2E8F0', alpha=0.9)
d2.set_facecolor('#FFFFFF')
d2.legend(frameon=True, facecolor='#F8FAFC', fontsize=9)

# Panel 3: Speedup Factor
d3.bar(['1 Thread', '2 Threads', '4 Threads'], speedup_advantage, color=[c_amber, c_slate, c_green], width=0.48, edgecolor='#334155', linewidth=1)
for i, val in enumerate(speedup_advantage):
    d3.annotate(f'{val:.2f}x Faster', xy=(i, val), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=c_dark)
d3.set_title('(C) Speedup Advantage: Padded vs Adjacent', fontsize=12, fontweight='bold', color=c_dark)
d3.set_ylabel('Speedup Factor (x)', fontsize=10.5, color=c_gray, fontweight='bold')
d3.set_ylim(0, 2.5)
d3.tick_params(axis='both', labelsize=9.5)
d3.grid(True, linestyle=':', color='#E2E8F0', alpha=0.9)
d3.set_facecolor('#FFFFFF')

# Panel 4: Memory Bandwidth Scaling
d4.bar(labels, sweep_bw, color=bar_colors, width=0.5, edgecolor='#334155', linewidth=1)
for i, bw in enumerate(sweep_bw):
    d4.annotate(f'{bw:.1f} GB/s', xy=(i, bw), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold', color=c_dark)
d4.set_title('(D) Memory Bandwidth: 512 MB Sweep (4 Passes)', fontsize=12, fontweight='bold', color=c_dark)
d4.set_ylabel('Bandwidth (GB/s)', fontsize=10.5, color=c_gray, fontweight='bold')
d4.set_ylim(0, 26)
d4.tick_params(axis='x', rotation=15, labelsize=9)
d4.tick_params(axis='y', labelsize=9.5)
d4.grid(True, linestyle=':', color='#E2E8F0', alpha=0.9)
d4.set_facecolor('#FFFFFF')

plt.suptitle('Parallel & GPU Computing Lab — Experiment 4: Coherence & NUMA Performance',
             fontsize=14, fontweight='bold', color=c_dark, y=0.995)
plt.tight_layout()
plt.savefig('images/performance_comparison_dashboard.png', dpi=200, bbox_inches='tight')
plt.close()

print('Experiment 4 charts regenerated with balanced sizing and high readability.')
