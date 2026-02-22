# Roofline Analysis: VLIW SIMD Tree-Traversal Hash Kernel

**Current optimized kernel (`perf_takehome.py`)**: **1172 cycles** on the submission benchmark (`forest_height=10`, `rounds=16`, `batch_size=256`) with the default settings in this repo.

The provided execution trace (`trace.json`, from the earlier 1189-cycle schedule) and the current static instruction mix both show the kernel is **compute-bound** (VALU/ALU) and within ~**20 cycles** of the pure throughput roofline for the current schedule.

## 1. Machine Architecture Summary

The target is a custom single-core VLIW (Very Long Instruction Word) SIMD processor with the following per-cycle resource limits:

| Engine | Slots/Cycle | Description |
|--------|-------------|-------------|
| **ALU** | 12 | Scalar 32-bit arithmetic/logic on scratch addresses |
| **VALU** | 6 | Vector ops on VLEN=8 element vectors (includes `multiply_add`) |
| **Load** | 2 | Scalar `load`, vector `vload` (8 contiguous), `const`, `load_offset` |
| **Store** | 2 | Scalar `store`, vector `vstore` (8 contiguous) |
| **Flow** | 1 | `select`/`vselect`, `cond_jump`, `add_imm`, `halt` |

Key properties:
- **VLEN = 8**: Each VALU op processes 8 elements simultaneously.
- **Scratch space = 1536 words**: Acts as the register file / constant pool / manual cache. All ALU/VALU operands are scratch addresses.
- **No indirect scratch access**: Instruction operands are fixed addresses determined at code-generation time. Indirect memory access requires a `load` instruction.
- **Write-after-read semantics**: Within a single cycle, all engines read inputs first, then all writes take effect at end of cycle. This means independent operations in the same bundle see the "old" values.
- **All engines execute simultaneously**: In a single cycle, we can issue up to 12 ALU + 6 VALU + 2 Load + 2 Store + 1 Flow ops concurrently, as long as slot limits are respected.

## 2. Workload Characterization

### Algorithm

The kernel performs a **batched parallel tree traversal with hashing**:

```
for round in range(16):           # 16 rounds
    for i in range(256):          # 256 batch elements
        idx = indices[i]
        val = values[i]
        node_val = tree[idx]       # data-dependent load
        val = myhash(val ^ node_val)
        idx = 2*idx + (1 if val%2==0 else 2)
        if idx >= n_nodes: idx = 0 # wrap at tree boundary
        indices[i] = idx
        values[i] = val
```

### Problem Parameters (submission test)

| Parameter | Value |
|-----------|-------|
| `forest_height` | 10 |
| `n_nodes` | 2047 (= 2^11 - 1) |
| `batch_size` | 256 |
| `rounds` | 16 |
| `VLEN` | 8 |
| SIMD groups | 32 (= 256 / 8) |

### Hash Function

The hash consists of 6 stages, each computing:
```
a = op2(op1(a, const1), op3(a, const3))
```

The stages are:
| Stage | op1 | const1 | op2 | op3 | const3 | Structure |
|-------|-----|--------|-----|-----|--------|-----------|
| 0 | + | 0x7ED55D16 | + | << | 12 | a = (a + C) + (a << 12) |
| 1 | ^ | 0xC761C23C | ^ | >> | 19 | a = (a ^ C) ^ (a >> 19) |
| 2 | + | 0x165667B1 | + | << | 5 | a = (a + C) + (a << 5) |
| 3 | + | 0xD3A2646C | ^ | << | 9 | a = (a + C) ^ (a << 9) |
| 4 | + | 0xFD7046C5 | + | << | 3 | a = (a + C) + (a << 3) |
| 5 | ^ | 0xB55A4F09 | ^ | >> | 16 | a = (a ^ C) ^ (a >> 16) |

## 3. Operation Count Analysis

### Starter baseline (for context): ~1 slot per instruction bundle

The original starter code issues one operation per cycle. Per element per round:

| Operation | Count | Engine |
|-----------|-------|--------|
| Address calculations | 6 | ALU |
| XOR val^node_val | 1 | ALU |
| Hash (6 stages × 3 ops) | 18 | ALU |
| Branch direction (%, ==, *, +) | 4 | ALU |
| Wrap check (<) | 1 | ALU |
| Load (idx, val, node_val) | 3 | Load |
| Store (idx, val) | 2 | Store |
| Select (branch, wrap) | 2 | Flow |
| **Total** | **37** | |

**Baseline total: 37 × 256 × 16 = 151,552 instructions ≈ 147,734 cycles** (close to reported baseline).

### Current optimized kernel (unrolled + list-scheduled VLIW)

The current kernel is a straight-line program (no runtime branching when `pause` is disabled in tests), so the cycle count equals the number of non-debug VLIW bundles.

Static instruction mix for the current best schedule (**1172 cycles**):

| Engine | Slot Demand | Slots/Cycle | Throughput Bound | Avg Utilization |
|--------|-------------|------------:|-----------------:|----------------:|
| **VALU** | 6,907 | 6 | **1,152** | **98.2%** |
| **ALU** | 13,682 | 12 | 1,141 | 97.3% |
| **Load** | 2,146 | 2 | 1,073 | 91.6% |
| **Flow** | 741 | 1 | 741 | 63.2% |
| **Store** | 64 | 2 | 32 | 2.7% |

So the pure throughput roofline is **1152 cycles** (VALU-bound). The measured **1172** is ~**1.7%** above this bound, indicating remaining headroom is mostly from **dependencies / scheduler suboptimality**, not unused engine capacity.

Notable op mix (counts across the full program):
- **VALU**: `multiply_add` 2016, `^` 2016, `>>` 1024, `+` 705, `&` 608, `<<` 512
- **ALU**: `^` 8704, `+` 4973
- **Load**: scattered `load` 2063 (2048 node loads + 15 tree preloads), `vload` 64 (idx/val), `const` 19
- **Flow**: `vselect` 704 (tree lookup for early levels), `add_imm` 35, `pause` 2

### Vectorized operation count (VLEN=8, 32 groups)

Per group of 8 elements per round (naive vectorization):

| Operation | Count | Engine | Slots |
|-----------|-------|--------|-------|
| vload idx | 1 | load | 1 |
| vload val | 1 | load | 1 |
| Scattered load node_val | 8 | load | 8 |
| valu XOR | 1 | valu | 1 |
| valu hash (6 × 3) | 18 | valu | 18 |
| valu branch calc | 4 | valu | 4 |
| valu wrap check | 2 | valu | 2 |
| vstore idx | 1 | store | 1 |
| vstore val | 1 | store | 1 |
| **Total** | | | **37** |

Address computation for scattered loads adds ~1 VALU op (vector add of `forest_values_p` to idx vector).

## 4. Resource Throughput Bounds ("Rooflines")

The throughput bound for each resource is: `ceil(total_demand / slots_per_cycle)`.

### 4a. Naive Vectorization (all 256 loads scattered)

**Per round (32 groups):**

| Resource | Demand/Round | Capacity/Cycle | Cycles/Round |
|----------|-------------|----------------|--------------|
| **Load** | 32×2 vloads + 32×8 scalar = 320 | 2 | **160** |
| **VALU** | 32 × 25 = 800 | 6 | **134** |
| **Flow** | 32 × 2 = 64 | 1 | **64** |
| **Store** | 32 × 2 = 64 | 2 | **32** |
| **ALU** | ~0 (all vectorized) | 12 | 0 |

**16-round totals:**

| Resource | Total Cycles | Bottleneck? |
|----------|-------------|-------------|
| **Load** | **2,560** | **YES** |
| **VALU** | 2,144 | Close second |
| **Flow** | 1,024 | |
| **Store** | 512 | |

**Naive vectorized throughput bound: ~2,560 cycles (load-bound).**

### 4b. With `multiply_add` optimization

The VALU `multiply_add(dest, a, b, c)` computes `dest[i] = a[i]*b[i] + c[i]` in a single slot.

Hash stages where op2 is `+` and op3 is `<<` can be collapsed:
- `(a + C) + (a << k)` = `a * (2^k + 1) + C` = `multiply_add(a, const_vec, C_vec)`

This applies to stages 0, 2, 4 (3 of 6 stages):

| Stage | Original VALU | With multiply_add |
|-------|-------------|-------------------|
| 0: (a+C) + (a<<12) | 3 ops, 2 cycles | **1 op, 1 cycle** |
| 1: (a^C) ^ (a>>19) | 3 ops, 2 cycles | 3 ops, 2 cycles |
| 2: (a+C) + (a<<5)  | 3 ops, 2 cycles | **1 op, 1 cycle** |
| 3: (a+C) ^ (a<<9)  | 3 ops, 2 cycles | 3 ops, 2 cycles |
| 4: (a+C) + (a<<3)  | 3 ops, 2 cycles | **1 op, 1 cycle** |
| 5: (a^C) ^ (a>>16) | 3 ops, 2 cycles | 3 ops, 2 cycles |
| **Total** | **18 ops, 12 cycles** | **12 ops, 9 cycles** |

**VALU savings: 33% fewer hash ops (18 → 12).**

### 4c. Flow engine: when to use `vselect` vs. VALU/ALU tricks

It’s possible to “eliminate flow” by expressing selection with arithmetic, but for the current optimized kernel:
- **Flow is not saturated** (≈63% utilized), so spending flow slots is often fine if it reduces VALU pressure.
- `vselect` is especially valuable for **tree lookup** at shallow levels once those node values are preloaded into scratch.

Arithmetic selection is still a useful tool. For example, branch direction can be computed without flow:
```
tmp = val & 1                          # valu &
tmp = tmp + 1                          # valu +   (gives 1 if even, 2 if odd)
idx = multiply_add(idx, 2, tmp)        # valu multiply_add  (2*idx + tmp)
```

For wrap: in this benchmark shape, wrap only occurs after visiting depth-10 nodes, and the current kernel handles it by unconditionally setting `idx = 0` on round 10.

### 4d. Used in the current kernel: preload levels 0–3 and `vselect` tree lookup

All elements start at index 0, so the first few rounds touch only a small, fixed set of nodes. The current kernel exploits this by:
- Preloading `tree[0..14]` into scratch once.
- Using flow `vselect` cascades at levels 1–3 to select the correct node value per lane without scattered loads.

For `rounds=16` with the level pattern `0,1,2,3,4,5,6,7,8,9,10,0,1,2,3,4`, this yields:
- **No scattered loads** for levels **0–3** (8 rounds total).
- **Scattered loads** only for levels **4–10 and 4** (8 rounds total) ⇒ `8 × 32 × 8 = 2048` node loads, plus 15 scalar tree preloads.

There are deeper structure-based load-sharing tricks (e.g., sharing loads when the number of reachable nodes is small), but since the kernel is now compute-bound, load reductions alone often don’t move the cycle count unless they also cut VALU/ALU work or shorten critical dependencies.

### 4e. Address formation: ALU vs VALU tradeoff (`ADDR_ALU_MASK`)

In scattered-load rounds (levels ≥ 4), forming `addr = idx + forest_values_p` can be done either as:
- **1 VALU** vector add, or
- **8 ALU** scalar adds (one per lane).

Because the current kernel is **VALU-bound**, it’s beneficial to shift selected address work onto ALU to relieve VALU pressure. The best default for this repo is:
- `ADDR_ALU_MASK` bits set for **levels 5, 6, 8**.

### 4f. Micro-optimization: reuse branch-history bits at level 2

At level 2, the node selection depends on bits of `(idx - 3)`. Those bits can be derived from previously-computed branch bits:
- `(idx - 3) & 2` is exactly the **level-0 branch bit** (`b0`), which can be kept live.
- `(idx - 3) & 1` is exactly the **level-1 branch bit** (`b1`), which can be stored in a temp vector and reused.

Reusing these bits avoids recomputing offset bits from `idx` and removes **2 VALU ops per group** for each level-2 round.

### 4g. Roofline for the current optimized kernel

Using the static instruction mix in §3, the pure throughput bound is:
`max(ceil(VALU/6), ceil(ALU/12), ceil(Load/2), ceil(Flow/1), ceil(Store/2))`
= `max(1152, 1141, 1073, 741, 32)` = **1152 cycles** (VALU-bound).

Measured: **1172 cycles**.

So the remaining ~20 cycles are attributable to **dependency-induced bubbles** and the fact that the greedy list scheduler does not always achieve the global optimum packing even when the aggregate demand suggests it might be possible.

## 5. Data Dependency and Latency Analysis

### Critical path per element per round

The serial dependency chain within one hash computation:

```
load node_val [depends on idx]
    → XOR val^node_val [depends on load]
        → hash stage 0 [1 cycle with multiply_add]
            → hash stage 1 [2 cycles: parallel pair + combine]
                → hash stage 2 [1 cycle]
                    → hash stage 3 [2 cycles]
                        → hash stage 4 [1 cycle]
                            → hash stage 5 [2 cycles]
                                → branch calc [3 cycles: &, +1, multiply_add]
```

**Critical path latency per element per round:**

| Phase | Cycles |
|-------|--------|
| Load node_val (scattered, 8 loads at 2/cycle for SIMD group) | 4 |
| XOR | 1 |
| Hash (9 cycles with multiply_add) | 9 |
| Branch calculation | 3 |
| **Total latency per group** | **17** |

### Cross-round dependency

Round N+1 depends on round N for the same element (both `idx` and `val` feed forward). This creates a serial chain:

**Minimum latency for one element across all 16 rounds: 17 × 16 = 272 cycles**

However, with 32 independent SIMD groups per round, we can pipeline: while group 0 finishes round N, groups 1-31 can still be working on round N, and group 0 of round N+1 can start as soon as group 0 of round N completes.

### Pipeline initiation interval

The initiation interval (II) is the minimum cycles between starting consecutive groups:

- **VALU throughput per group**: ~17 ops / 6 slots ≈ 3 cycles
- **Load throughput per group**: 8 scattered loads / 2 slots = 4 cycles (for scattered rounds)

**II ≈ 4 cycles** for scattered rounds, **3 cycles** for efficient rounds.

With 32 groups per round: steady-state cycles per round ≈ II × 32 + pipeline_drain
- Scattered: 4 × 32 + 17 ≈ 145 cycles
- Efficient: 3 × 32 + 17 ≈ 113 cycles

But with cross-round pipelining (overlapping end of round N with start of round N+1), the drain cost is amortized.

## 6. Summary: Throughput Bound Table

The earlier rows are rough “path-to-optimization” estimates for simplified kernels; the final row reflects the **actual** instruction mix of the current optimized kernel in this repo.

| Optimization Level | VALU Bound | Load Bound | Flow Bound | Estimated Cycles |
|-------------------|-----------|-----------|-----------|-----------------|
| **Baseline** (scalar, 1 op/cycle) | — | — | — | **147,734** |
| Naive VLIW packing (scalar) | — | — | — | ~18,000 |
| **Vectorized** (naive) | 2,144 | **2,560** | 1,024 | **~2,560** |
| + multiply_add | 1,536 | **2,560** | 1,024 | **~2,560** |
| + eliminate flow ops | 1,536 | **2,560** | 0 | **~2,560** |
| + shared-round loads | **1,536** | 527 | 0 | **~1,536** |
| + skip wrap for safe rounds | **1,296** | 527 | 0 | **~1,350** |
| **Current optimized kernel (this repo)** | **1,152** | 1,073 | 741 | **1,172** |

## 7. Theoretical Lower Bounds

### Pure throughput lower bound

For the current optimized kernel’s instruction mix, the throughput roofline is **1152 cycles** (VALU-bound). The measured result (**1172 cycles**) is close enough that further improvements will likely require:
- reducing total **VALU** demand (or shifting work to ALU without creating bubbles), and/or
- reshaping dependencies so the scheduler can pack closer to the roofline.

### Absolute lower bound

If we could somehow reduce VALU demand further (e.g., discovering algebraic simplifications in the hash):
- Load-bound: 527 cycles (with shared-round optimization)
- This would be the floor if VALU could be reduced enough

### Comparison with benchmarks

| Target | Cycles | Bottleneck |
|--------|--------|-----------|
| Baseline | 147,734 | No parallelism |
| VLIW scalar | ~18,532 | ALU throughput |
| Vectorized + basic opts | ~2,164 | Load bandwidth |
| Current optimized kernel (this repo) | **1,172** | VALU/ALU throughput |
| Theoretical load floor | ~527 | — |

## 8. Absolute Minimum Cycles (what’s the floor?)

For the submission benchmark shape (`forest_height=10`, `rounds=16`, `batch_size=256`, `VLEN=8`), the best notion of “absolute minimum cycles” is a **roofline lower bound**: even with perfect scheduling and no bubbles, the ISA’s per-engine slot limits cap how much work can be issued per cycle.

### 8a. A hard-ish VALU floor (functionally unavoidable)

The hash has 6 stages; with this ISA the best-known decomposition is:
- 3 affine stages (`(a + C) + (a << k)`) ⇒ **1× `multiply_add`** each
- 3 XOR/shift stages (`(a ⊕ C) ⊕ (a >> k)` and `(a + C) ⊕ (a << k)`) ⇒ at least **1 shift + 1 XOR-combine** each, with constant-mixing pushed to ALU

That gives a lower bound of **9 VALU ops / round / group** for hashing.

For the rest of the loop:
- **Node mix** needs an XOR with the selected/loaded node value every round; because the submission inputs start with `idx=0` and round 11 wraps back to `idx=0`, the two level-0 rounds always use the scalar `tree[0]` and can be done as scalar ALU XORs, so the best-case floor is **14 VALU XORs / group** (not 16).
- **Index update** for 15 rounds needs: `b = val & 1`, `idx = 2*idx + 1`, `idx += b` ⇒ **3 VALU ops / round**. Round 10 always wraps, so it can be a single “set zero” op.

Putting those together:
- Hash: `9 × 16 = 144` VALU ops / group
- Node XOR: `14` VALU ops / group
- Index update: `15 × 3 + 1 = 46` VALU ops / group
- **Total (floor): `144 + 14 + 46 = 204` VALU ops / group**

Across 32 groups: `204 × 32 = 6528` VALU ops ⇒ with 6 VALU slots/cycle:
`ceil(6528 / 6) = 1088` cycles.

So **1088 cycles** is a strong candidate for the absolute floor *if* you can eliminate essentially all other VALU overhead (e.g., level-3 condition formation and scattered address formation) without increasing the critical path.

### 8b. Multi-engine feasibility at that floor

At 1088 cycles, the per-engine budgets are:
- ALU: `1088 × 12 = 13056` slots
- VALU: `1088 × 6 = 6528` slots
- Load: `1088 × 2 = 2176` slots
- Flow: `1088 × 1 = 1088` slots

The current optimized kernel uses:
- VALU: 6907 ops (needs ~379 fewer to fit in 6528)
- ALU: 13682 ops (needs ~626 fewer to fit in 13056)

Those gaps line up with the obvious remaining “real work” above the floor:
- scattered address formation (currently split across VALU and ALU depending on `ADDR_ALU_MASK`)
- level-3 condition formation (`idx+1`, masks) for `vselect` tree lookup

### 8c. My best guess

Given the ISA (no gather, 1-cycle ops, limited scratch) and how close the current kernel already is to its **1152-cycle** VALU roofline, I’d expect the true optimum for **fully correct values + indices** on this benchmark to be in the neighborhood of **1090–1120 cycles**, with **1088** as the most plausible absolute floor.

## 9. Key Optimization Priorities (by impact)

1. **Vectorize with VLEN=8** (~8× speedup potential): Convert scalar ops to VALU, use vload/vstore for contiguous data.

2. **VLIW packing** (~8× additional): Pack independent operations into the same instruction bundle to utilize all engine slots per cycle.

3. **`multiply_add` for hash stages 0, 2, 4** (33% hash reduction): Collapse `(a+C)+(a<<k)` = `a*(2^k+1)+C` into a single VALU slot.

4. **Use flow strategically**: Spend flow `vselect` slots when it reduces memory traffic or VALU pressure; don’t “eliminate flow” if it forces extra VALU ops.

5. **Exploit tree traversal structure** (3.9× load reduction): Share node_val loads across elements at the same tree level in early rounds; recognize the wrap-to-zero pattern at level 10.

6. **Skip wrap check for safe rounds** (further VALU reduction): Only round 10 needs wrapping; all other rounds are guaranteed in-bounds.

7. **Software pipelining across groups** (hide latency): Overlap loads for one group with hash computation for another to keep all engines busy simultaneously.

8. **Keep idx/val in scratch across rounds** (eliminate redundant loads/stores): Only load from memory at the start and store at the end, not every round.

9. **Bit reuse across levels** (shave VALU ops): Preserve and reuse branch-history bits instead of recomputing offset bits from `idx` (e.g., level-2 selection).

10. **Balance VALU vs ALU pressure** (improve packability): Move selected address-formation work from VALU to ALU where it reduces the VALU roofline (`ADDR_ALU_MASK`).
