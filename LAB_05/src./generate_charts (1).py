import os
import matplotlib.pyplot as plt
import numpy as np

# Create output directories if needed
os.makedirs('images', exist_ok=True)

# Set custom aesthetic typography & style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Segoe UI', 'Arial']

# Empirical measurements from Experiment 5 run on NVIDIA RTX 5060 Ti
cpu_time = 1.154830          # ms
gpu_kernel_time = 0.027536   # ms
gpu_e2e_time = 0.114065      # ms
transfer_time = gpu_e2e_time - gpu_kernel_time # 0.086529 ms

kernel_speedup = 41.938918   # x
e2e_speedup = 10.124315      # x

# Color palette
colors = ['#E17055', '#0984E3', '#00B894']
speedup_colors = ['#00B894', '#0984E3']

# ============================================================================
# 1. Comprehensive 3-Panel Benchmark Dashboard
# ============================================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.8), facecolor='#FAFAFA')

# Panel 1: Execution Time (Log Scale)
categories = ['CPU Forward Pass\n(Single-Core C++)', 'GPU End-to-End\n(Compute + PCIe I/O)', 'GPU Kernel-Only\n(CUDA Cores)']
times = [cpu_time, gpu_e2e_time, gpu_kernel_time]
bars1 = ax1.bar(categories, times, color=colors, width=0.52, edgecolor='#2D3436', linewidth=1.5)
ax1.set_yscale('log')
ax1.set_title('Execution Time per Forward Pass (Log Scale)', fontsize=13, fontweight='bold', color='#2D3436', pad=12)
ax1.set_ylabel('Execution Time in Milliseconds (ms)', fontsize=11, fontweight='bold', color='#636E72')
ax1.grid(True, which='both', linestyle=':', color='#DFE6E9', alpha=0.8)
ax1.set_facecolor('#FFFFFF')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

for bar, t in zip(bars1, times):
    yval = bar.get_height()
    label = f'{t:.3f} ms' if t >= 0.1 else f'{t:.4f} ms'
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval * 1.35, label,
             ha='center', va='bottom', fontsize=10.5, fontweight='bold', color='#2D3436',
             bbox=dict(boxstyle='round,pad=0.3', fc='#F1F2F6', ec='#B2BEC3', lw=1))

# Panel 2: Speedup Comparison
speedup_cats = ['Kernel-Only Speedup\n(Compute Scaling)', 'End-to-End Speedup\n(System Throughput)']
speedups = [kernel_speedup, e2e_speedup]
bars2 = ax2.bar(speedup_cats, speedups, color=speedup_colors, width=0.48, edgecolor='#2D3436', linewidth=1.5)
ax2.set_title('Measured Speedup vs. CPU Baseline', fontsize=13, fontweight='bold', color='#2D3436', pad=12)
ax2.set_ylabel('Speedup Factor (x, Higher is Better)', fontsize=11, fontweight='bold', color='#636E72')
ax2.set_ylim(0, 50)
ax2.grid(True, linestyle=':', color='#DFE6E9', alpha=0.8)
ax2.set_facecolor('#FFFFFF')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

for bar, s in zip(bars2, speedups):
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f'{s:.2f}x',
             ha='center', va='bottom', fontsize=11, fontweight='bold', color='#2D3436',
             bbox=dict(boxstyle='round,pad=0.3', fc='#F1F2F6', ec='#B2BEC3', lw=1))

# Panel 3: GPU Latency Breakdown (PCIe Transfers vs Compute)
kernel_pct = (gpu_kernel_time / gpu_e2e_time) * 100
transfer_pct = (transfer_time / gpu_e2e_time) * 100

breakdown_labels = [f'PCIe Host-Device Transfers\n({transfer_pct:.1f}%)', f'CUDA Kernel Compute\n({kernel_pct:.1f}%)']
breakdown_sizes = [transfer_time, gpu_kernel_time]
breakdown_colors = ['#6C5CE7', '#00B894']

wedges, texts, autotexts = ax3.pie(
    breakdown_sizes, labels=breakdown_labels, autopct='%1.1f%%',
    startangle=140, colors=breakdown_colors,
    wedgeprops=dict(edgecolor='#2D3436', linewidth=1.5),
    textprops=dict(fontweight='bold', color='#2D3436')
)
for at in autotexts:
    at.set_color('#FFFFFF')
    at.set_fontsize(11)
ax3.set_title('GPU End-to-End Latency Breakdown\n(PCIe I/O vs. Core Execution)', fontsize=13, fontweight='bold', color='#2D3436', pad=12)

plt.suptitle('Parallel & GPU Computing Lab — Experiment 5: Neural Network Layers in CUDA Benchmark',
             fontsize=15, fontweight='bold', color='#2D3436', y=1.02)
plt.tight_layout()
plt.savefig('images/performance_comparison_dashboard.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 2. Standalone Execution Time Comparison
# ============================================================================
fig, ax = plt.subplots(figsize=(9, 5.5), facecolor='#FAFAFA')
bars = ax.bar(categories, times, color=colors, width=0.48, edgecolor='#2D3436', linewidth=1.5)
ax.set_yscale('log')
ax.set_title('Two-Layer Neural Network Forward Pass Execution Latency', fontsize=14, fontweight='bold', color='#2D3436', pad=15)
ax.set_ylabel('Execution Time per Sample Batch (ms, Log Scale)', fontsize=11, fontweight='bold', color='#636E72')
ax.grid(True, which='both', linestyle=':', color='#DFE6E9', alpha=0.8)
ax.set_facecolor('#FFFFFF')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for bar, t in zip(bars, times):
    yval = bar.get_height()
    label = f'{t:.4f} ms' if t < 0.1 else f'{t:.3f} ms'
    ax.text(bar.get_x() + bar.get_width()/2.0, yval * 1.3, label,
            ha='center', va='bottom', fontsize=11, fontweight='bold', color='#2D3436',
            bbox=dict(boxstyle='round,pad=0.3', fc='#F1F2F6', ec='#B2BEC3', lw=1))

plt.tight_layout()
plt.savefig('images/execution_time_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 3. Standalone Speedup Analysis
# ============================================================================
fig, ax = plt.subplots(figsize=(8.5, 5.5), facecolor='#FAFAFA')
bars = ax.bar(speedup_cats, speedups, color=speedup_colors, width=0.45, edgecolor='#2D3436', linewidth=1.5)
ax.set_title('CUDA GPU Acceleration Speedup over Single-Threaded CPU', fontsize=14, fontweight='bold', color='#2D3436', pad=15)
ax.set_ylabel('Speedup Factor vs. CPU Baseline (x)', fontsize=11, fontweight='bold', color='#636E72')
ax.set_ylim(0, 50)
ax.grid(True, linestyle=':', color='#DFE6E9', alpha=0.8)
ax.set_facecolor('#FFFFFF')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for bar, s in zip(bars, speedups):
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f'{s:.2f}x',
            ha='center', va='bottom', fontsize=11.5, fontweight='bold', color='#2D3436',
            bbox=dict(boxstyle='round,pad=0.3', fc='#F1F2F6', ec='#B2BEC3', lw=1))

ax.axhline(1.0, color='#D63031', linestyle='--', linewidth=1.5, label='Sequential CPU Baseline (1.0x)')
ax.legend(frameon=True, facecolor='#FFFFFF', edgecolor='#DFE6E9', fontsize=10.5)

plt.tight_layout()
plt.savefig('images/speedup_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print('Experiment 5 charts generated successfully.')
