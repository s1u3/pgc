#!/usr/bin/env python3
"""
coherence_sim.py - Cache Coherence Protocol Simulator (MSI, MESI, MOESI)
Course: Parallel and GPU Computing (26ECAC304) | Experiment 4

Simulates a 2-core snooping cache architecture tracking state transitions
for MSI, MESI, and MOESI protocols under adjacent and padded memory layouts.
Workload: Core 0 increments counter X (Read, Write), Core 1 increments counter Y (Read, Write).
- Adjacent layout: X and Y share Cache Line A (False Sharing ping-pong).
- Padded layout: X on Line A, Y on Line B (Independent cache lines).
"""

import sys

def make_cores(n=2):
    return [dict() for _ in range(n)]

def access(proto, caches, core, line, op, st):
    """
    Simulates a memory access (op = 'R' or 'W') by a specific core on a cache line.
    Updates cache states according to the protocol rules and increments transaction statistics.
    """
    me = caches[core]
    s = me.get(line, 'I')
    others = [i for i in range(len(caches)) if i != core]
    holders = [i for i in others if caches[i].get(line, 'I') != 'I']

    if op == 'R':
        if s != 'I':
            return  # Read hit, no bus action required
        st['BusRd'] += 1
        new = 'S'
        for i in holders:
            o = caches[i][line]
            if o == 'M':
                if proto == 'MOESI':
                    caches[i][line] = 'O'
                    st['c2c'] += 1  # Owner supplies data via cache-to-cache transfer (no memory write-back)
                else:
                    caches[i][line] = 'S'
                    st['writeback'] += 1  # Dirty line must be flushed to main memory
            elif o == 'O':
                st['c2c'] += 1  # Owner continues to supply data
            elif o == 'E':
                caches[i][line] = 'S'
        if not holders and proto in ('MESI', 'MOESI'):
            new = 'E'  # Clean, exclusive ownership
        me[line] = new
    else:  # op == 'W'
        if s == 'M':
            return  # Write hit in Modified state, zero bus transaction
        if s == 'E':
            me[line] = 'M'
            st['silent E->M'] += 1  # Silent upgrade without bus transaction (MESI / MOESI advantage)
            return
        if s in ('S', 'O'):
            st['BusUpgr'] += 1  # Broadcast BusUpgr to invalidate other shared copies
        else:  # s == 'I'
            st['BusRdX'] += 1  # Read-for-Ownership on write miss
            for i in holders:
                o = caches[i][line]
                if o in ('M', 'O'):
                    if proto == 'MOESI':
                        st['c2c'] += 1
                    else:
                        st['writeback'] += 1
        for i in holders:
            caches[i][line] = 'I'
            st['invalidations'] += 1
        me[line] = 'M'

def run(proto, layout, iters):
    st = dict.fromkeys(['BusRd', 'BusRdX', 'BusUpgr', 'silent E->M', 'invalidations', 'writeback', 'c2c'], 0)
    caches = make_cores()
    lx, ly = ('A', 'A') if layout == 'adjacent' else ('A', 'B')
    trace = []
    for _ in range(iters):
        trace += [(0, lx, 'R'), (0, lx, 'W'), (1, ly, 'R'), (1, ly, 'W')]
    
    log = []
    for core, line, op in trace:
        access(proto, caches, core, line, op, st)
        log.append((core, line, op, caches[0].get('A', 'I'), caches[1].get('A', 'I')))
    
    st['bus transactions'] = st['BusRd'] + st['BusRdX'] + st['BusUpgr']
    return st, log

if __name__ == '__main__':
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    print("=" * 70)
    print("CACHE COHERENCE PROTOCOL SIMULATOR (MSI vs MESI vs MOESI)")
    print("=" * 70)
    print("\n[PART A] State Trace: First 2 Iterations (Adjacent Layout, Line A State: C0 / C1)")
    print("-" * 70)
    
    for proto in ('MSI', 'MESI', 'MOESI'):
        _, log = run(proto, 'adjacent', 2)
        print(f"\nProtocol: {proto}")
        for c, l, o, s0, s1 in log:
            op_desc = "Reads " if o == 'R' else "Writes"
            print(f"  Core {c} {op_desc} line {l} -> State: [C0={s0}, C1={s1}]")

    print("\n" + "=" * 70)
    print(f"[PART B] Aggregate Coherence Statistics for {iters} Iterations ({iters * 4} Operations)")
    print("=" * 70)
    keys = ['bus transactions', 'BusRd', 'BusRdX', 'BusUpgr', 'invalidations', 'writeback', 'c2c', 'silent E->M']
    header = f"{'Layout':9} {'Proto':6} " + " ".join(f"{k:>14}" for k in keys)
    print(header)
    print("-" * len(header))
    for layout in ('adjacent', 'padded'):
        for proto in ('MSI', 'MESI', 'MOESI'):
            st, _ = run(proto, layout, iters)
            print(f"{layout:9} {proto:6} " + " ".join(f"{st[k]:>14}" for k in keys))
    print("-" * len(header))
