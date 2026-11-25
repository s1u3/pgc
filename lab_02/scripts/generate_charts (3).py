import os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs('images', exist_ok=True)

# Typography & theme settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Segoe UI', 'Arial']

threads = [1, 2, 4, 6, 16]
seq_time = 1.418018

pthread_times = [1.405171, 0.718377, 0.358913, 0.240157, 0.136414]
omp_times = [1.393317, 0.717785, 0.359875, 0.240754, 0.136195]

pthread_speedup = [1.009, 1.974, 3.951, 5.905, 10.395]
omp_speedup = [1.018, 1.976, 3.940, 5.890, 10.412]
ideal_speedup = [1, 2, 4, 6, 16]

pthread_eff = [100.91, 98.70, 98.77, 98.41, 64.97]
omp_eff = [101.77, 98.78, 98.51, 98.16, 65.07]

# Muted, professional engineering palette
c_pthread = '#2B5C8F'  # Deep Slate Blue
c_omp = '#387864'      # Muted Teal / Forest
c_ideal = '#94A3B8'    # Soft Slate Gray
c_seq = '#B85D43'      # Muted Terracotta

# -------------------------------------------------------------
# 1. Combined 3-Panel Benchmark Overview Chart
# -------------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 4.8), facecolor='#FAFBFC')

# Panel 1: Execution Time
ax1.plot(threads, pthread_times, marker='o', markersize=7, color=c_pthread, lw=2.4, label='Pthreads')
ax1.plot(threads, omp_times, marker='s', markersize=7, color=c_omp, lw=2.4, linestyle='--', label='OpenMP')
ax1.axhline(seq_time, color=c_seq, linestyle=':', lw=1.8, label=f'Sequential ({seq_time:.2f}s)')
ax1.set_title('Execution Time vs. Threads', fontsize=12.5, fontweight='bold', color='#1E293B', pad=12)
ax1.set_xlabel('Thread Count (P)', fontsize=11, color='#334155', fontweight='bold')
ax1.set_ylabel('Execution Time in Seconds', fontsize=11, color='#334155', fontweight='bold')
ax1.set_xticks(threads)
ax1.tick_params(axis='both', labelsize=10)
ax1.grid(True, linestyle=':', color='#E2E8F0', alpha=0.9)
ax1.set_facecolor('#FFFFFF')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.legend(frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1', fontsize=9.5)

# Panel 2: Speedup
ax2.plot(threads, pthread_speedup, marker='o', markersize=7, color=c_pthread, lw=2.4, label='Pthreads')
ax2.plot(threads, omp_speedup, marker='s', markersize=7, color=c_omp, lw=2.4, linestyle='--', label='OpenMP')
ax2.plot(threads, ideal_speedup, color=c_ideal, linestyle=':', lw=1.8, label='Ideal Linear (P)')
ax2.set_title('Speedup vs. Threads', fontsize=12.5, fontweight='bold', color='#1E293B', pad=12)
ax2.set_xlabel('Thread Count (P)', fontsize=11, color='#334155', fontweight='bold')
ax2.set_ylabel('Speedup Factor (x)', fontsize=11, color='#334155', fontweight='bold')
ax2.set_xticks(threads)
ax2.tick_params(axis='both', labelsize=10)
ax2.grid(True, linestyle=':', color='#E2E8F0', alpha=0.9)
ax2.set_facecolor('#FFFFFF')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.legend(frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1', fontsize=9.5)

# Panel 3: Parallel Efficiency
ax3.plot(threads, pthread_eff, marker='o', markersize=7, color=c_pthread, lw=2.4, label='Pthreads')
ax3.plot(threads, omp_eff, marker='s', markersize=7, color=c_omp, lw=2.4, linestyle='--', label='OpenMP')
ax3.axhline(100, color=c_ideal, linestyle=':', lw=1.8, label='100% Ideal Efficiency')
ax3.set_title('Parallel Efficiency (%) vs. Threads', fontsize=12.5, fontweight='bold', color='#1E293B', pad=12)
ax3.set_xlabel('Thread Count (P)', fontsize=11, color='#334155', fontweight='bold')
ax3.set_ylabel('Efficiency (%)', fontsize=11, color='#334155', fontweight='bold')
ax3.set_xticks(threads)
ax3.set_ylim(50, 112)
ax3.tick_params(axis='both', labelsize=10)
ax3.grid(True, linestyle=':', color='#E2E8F0', alpha=0.9)
ax3.set_facecolor('#FFFFFF')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.legend(frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1', fontsize=9.5)

plt.tight_layout()
plt.savefig('images/performance_comparison_charts.png', dpi=200, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# 2. Standalone Execution Time Chart
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5), facecolor='#FAFBFC')
ax.plot(threads, pthread_times, marker='o', markersize=8, color=c_pthread, lw=2.5, label='Pthreads')
ax.plot(threads, omp_times, marker='s', markersize=8, color=c_omp, lw=2.5, linestyle='--', label='OpenMP')
ax.axhline(seq_time, color=c_seq, linestyle=':', lw=1.8, label=f'Sequential Baseline ({seq_time:.2f}s)')

for t, pt in zip(threads, pthread_times):
    ax.text(t, pt + 0.04, f'{pt:.3f}s', ha='center', fontsize=9.5, fontweight='bold', color='#1E293B')

ax.set_title('Execution Time Scaling (Lower is Better)', fontsize=13, fontweight='bold', color='#1E293B', pad=12)
ax.set_xlabel('Thread Count (P)', fontsize=11, color='#334155', fontweight='bold')
ax.set_ylabel('Execution Time in Seconds', fontsize=11, color='#334155', fontweight='bold')
ax.set_xticks(threads)
ax.tick_params(axis='both', labelsize=10)
ax.grid(True, linestyle=':', color='#E2E8F0', alpha=0.9)
ax.set_facecolor('#FFFFFF')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1', fontsize=10)

plt.tight_layout()
plt.savefig('images/execution_time_vs_threads.png', dpi=200, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# 3. Standalone Speedup Chart
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5), facecolor='#FAFBFC')
ax.plot(threads, ideal_speedup, color=c_ideal, linestyle=':', lw=1.8, label='Ideal Linear (P)')
ax.plot(threads, pthread_speedup, marker='o', markersize=8, color=c_pthread, lw=2.5, label='Pthreads')
ax.plot(threads, omp_speedup, marker='s', markersize=8, color=c_omp, lw=2.5, linestyle='--', label='OpenMP')

for t, ps in zip(threads, pthread_speedup):
    ax.text(t, ps + 0.35, f'{ps:.2f}x', ha='center', fontsize=9.5, fontweight='bold', color='#1E293B')

ax.set_title('Parallel Speedup vs. Number of Threads', fontsize=13, fontweight='bold', color='#1E293B', pad=12)
ax.set_xlabel('Thread Count (P)', fontsize=11, color='#334155', fontweight='bold')
ax.set_ylabel('Speedup Factor (x)', fontsize=11, color='#334155', fontweight='bold')
ax.set_xticks(threads)
ax.tick_params(axis='both', labelsize=10)
ax.grid(True, linestyle=':', color='#E2E8F0', alpha=0.9)
ax.set_facecolor('#FFFFFF')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1', fontsize=10)

plt.tight_layout()
plt.savefig('images/speedup_vs_threads.png', dpi=200, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# 4. Standalone Efficiency Chart
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5), facecolor='#FAFBFC')
ax.axhline(100, color=c_ideal, linestyle=':', lw=1.8, label='100% Ideal Efficiency')
ax.plot(threads, pthread_eff, marker='o', markersize=8, color=c_pthread, lw=2.5, label='Pthreads')
ax.plot(threads, omp_eff, marker='s', markersize=8, color=c_omp, lw=2.5, linestyle='--', label='OpenMP')

for t, pe in zip(threads, pthread_eff):
    offset = 2.5 if t != 16 else -5
    ax.text(t, pe + offset, f'{pe:.1f}%', ha='center', fontsize=9.5, fontweight='bold', color='#1E293B')

ax.set_title('Parallel Efficiency (%) vs. Number of Threads', fontsize=13, fontweight='bold', color='#1E293B', pad=12)
ax.set_xlabel('Thread Count (P)', fontsize=11, color='#334155', fontweight='bold')
ax.set_ylabel('Parallel Efficiency (%)', fontsize=11, color='#334155', fontweight='bold')
ax.set_xticks(threads)
ax.set_ylim(50, 112)
ax.tick_params(axis='both', labelsize=10)
ax.grid(True, linestyle=':', color='#E2E8F0', alpha=0.9)
ax.set_facecolor('#FFFFFF')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1', fontsize=10)

plt.tight_layout()
plt.savefig('images/efficiency_vs_threads.png', dpi=200, bbox_inches='tight')
plt.close()

print('Experiment 2 charts regenerated with balanced sizing and high readability.')
