# Roofline Analysis: VLIW SIMD Tree-Traversal Hash Kernel

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

### Baseline (current code): 1 slot per instruction bundle

The current code issues one operation per cycle. Per element per round:

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

### 4c. Eliminating flow ops with ALU tricks

The branch direction and wrap check can be converted from flow ops to pure VALU:

**Branch direction** (original uses `%`, `==`, flow `select`):
```
# Original: 2 ALU + 1 flow select
tmp = val % 2; cond = (tmp == 0); offset = select(cond, 1, 2)

# Optimized: pure VALU
tmp = val & 1                          # valu &
tmp = tmp + 1                          # valu +   (gives 1 if even, 2 if odd)
idx = multiply_add(idx, 2, tmp)        # valu multiply_add  (2*idx + tmp)
```
3 VALU ops, 0 flow ops.

**Wrap check** (original uses `<`, flow `select`):
```
# Original: 1 ALU + 1 flow select
cond = (idx < n_nodes); idx = select(cond, idx, 0)

# Optimized: pure VALU
cond = idx < n_nodes                   # valu <
idx = idx * cond                       # valu *   (0 if cond=0, idx if cond=1)
```
2 VALU ops, 0 flow ops.

**Result: Flow engine completely eliminated from inner loop.** Flow is no longer a bottleneck (0 flow ops vs. 1 slot/cycle).

### 4d. Exploiting tree traversal structure

All 256 elements start at index 0. The tree has height 10, so elements follow a structured traversal:

| Round | Tree Level Accessed | Max Unique Nodes | Loads Needed |
|-------|-------------------|-------------------|-------------|
| 0 | 0 | 1 | 1 |
| 1 | 1 | 2 | 2 |
| 2 | 2 | 4 | 4 |
| 3 | 3 | 8 | 8 |
| 4 | 4 | 16 | 16 |
| 5 | 5 | 32 | 32 |
| 6 | 6 | 64 | 64 |
| 7 | 7 | 128 | 128 |
| 8 | 8 | 256 | ≤256 |
| 9 | 9 | 256 | ≤256 |
| 10 | 10 | 256 | ≤256 |
| 11 | 0 (wrapped) | 1 | 1 |
| 12 | 1 | 2 | 2 |
| 13 | 2 | 4 | 4 |
| 14 | 3 | 8 | 8 |
| 15 | 4 | 16 | 16 |

**Key insight**: After round 10, ALL elements wrap to index 0 (because any child of a level-10 node exceeds `n_nodes = 2047`). The traversal pattern repeats from round 11.

**Total loads: 1+2+4+8+16+32+64+128+256+256+256+1+2+4+8+16 = 1,054 loads**

Compared to 4,096 scattered loads without this optimization (256 × 16), this is a **3.9× reduction**.

At 2 loads/cycle: **527 cycles** (no longer the bottleneck).

### 4e. Eliminating wrap computation for known-safe rounds

Since elements follow the tree level-by-level, wrapping only occurs after round 10 (from level 10). For all other rounds, `2*idx + 2 < n_nodes` is guaranteed, so the wrap check can be skipped:

| Rounds | Wrap needed? | Branch+Wrap VALU ops |
|--------|-------------|---------------------|
| 0-9, 11-15 (15 rounds) | No | 3 (branch only) |
| 10 | Always wraps to 0 | 1 (just broadcast zero) |

### 4f. Fully Optimized VALU Throughput Bound

Per group per round (with all optimizations above):

| Round Type | XOR | Hash | Branch | Wrap | Addr | Total VALU |
|-----------|-----|------|--------|------|------|-----------|
| Normal (15 rounds) | 1 | 12 | 3 | 0 | ~1 | **17** |
| Round 10 (wrap-to-zero) | 1 | 12 | 0 | 1 | ~1 | **15** |

**Total VALU ops: 32 × (15 × 17 + 1 × 15) = 32 × 270 = 8,640**

Hmm, let me recount more carefully. For round 10, we still need to load node_val and hash to update val, but idx is unconditionally set to 0:
- Round 10: XOR(1) + hash(12) + set_zero(1) + addr_calc(1) = 15 ops/group
- Other 15 rounds: XOR(1) + hash(12) + branch(3) + addr_calc(1) = 17 ops/group

Total: 32 × (15 × 17 + 1 × 15) = 32 × 270 = **8,640 VALU ops**

At 6 VALU slots/cycle: **8,640 / 6 = 1,440 cycles**

Removing addr_calc for early rounds where loads are broadcast (rounds 0-5, 11-15 → 10 rounds with ≤32 unique nodes where addresses are compile-time known):

- 10 "efficient" rounds: 32 × (1 + 12 + 3) = 512 VALU ops/round (no addr calc needed)
- 5 "scattered" rounds (6-10): as above with addr calc
  - Rounds 6-9: 32 × 17 = 544/round × 4 = 2,176
  - Round 10: 32 × 15 = 480

Total: 10 × 512 + 2,176 + 480 = 5,120 + 2,176 + 480 = **7,776 VALU ops**

At 6/cycle: **7,776 / 6 = 1,296 cycles**

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

| Optimization Level | VALU Bound | Load Bound | Flow Bound | Estimated Cycles |
|-------------------|-----------|-----------|-----------|-----------------|
| **Baseline** (scalar, 1 op/cycle) | — | — | — | **147,734** |
| Naive VLIW packing (scalar) | — | — | — | ~18,000 |
| **Vectorized** (naive) | 2,144 | **2,560** | 1,024 | **~2,560** |
| + multiply_add | 1,536 | **2,560** | 1,024 | **~2,560** |
| + eliminate flow ops | 1,536 | **2,560** | 0 | **~2,560** |
| + shared-round loads | **1,536** | 527 | 0 | **~1,536** |
| + skip wrap for safe rounds | **1,296** | 527 | 0 | **~1,350** |

## 7. Theoretical Lower Bounds

### Pure throughput lower bound

With all identified optimizations: **~1,296 cycles** (VALU-bound).

Adding realistic overhead (initial loads, final stores, constant setup, pipeline fill/drain):
- Setup: ~30-50 cycles
- Per-round pipeline gaps: ~5 × 16 = 80 cycles

**Estimated achievable: ~1,350-1,400 cycles**

This aligns closely with the best reported result of **1,363 cycles** (Claude Opus 4.5 in improved harness).

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
| All optimizations | ~1,363 | VALU throughput |
| Theoretical load floor | ~527 | — |

## 8. Key Optimization Priorities (by impact)

1. **Vectorize with VLEN=8** (~8× speedup potential): Convert scalar ops to VALU, use vload/vstore for contiguous data.

2. **VLIW packing** (~8× additional): Pack independent operations into the same instruction bundle to utilize all engine slots per cycle.

3. **`multiply_add` for hash stages 0, 2, 4** (33% hash reduction): Collapse `(a+C)+(a<<k)` = `a*(2^k+1)+C` into a single VALU slot.

4. **Eliminate flow ops** (remove flow bottleneck): Replace `select` with VALU arithmetic (`&`, `*`), freeing the flow engine entirely.

5. **Exploit tree traversal structure** (3.9× load reduction): Share node_val loads across elements at the same tree level in early rounds; recognize the wrap-to-zero pattern at level 10.

6. **Skip wrap check for safe rounds** (further VALU reduction): Only round 10 needs wrapping; all other rounds are guaranteed in-bounds.

7. **Software pipelining across groups** (hide latency): Overlap loads for one group with hash computation for another to keep all engines busy simultaneously.

8. **Keep idx/val in scratch across rounds** (eliminate redundant loads/stores): Only load from memory at the start and store at the end, not every round.
