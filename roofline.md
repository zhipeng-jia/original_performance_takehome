# Roofline-Style Cycle Lower Bound Analysis (This Takehome)

This note captures a back-of-the-envelope "roofline" for the kernel in this repo:
`reference_kernel2` in `problem.py` / `KernelBuilder.build_kernel` in `perf_takehome.py`.

The goal is to estimate a *theoretical* minimum cycle count given the simulated
machine's per-cycle engine slot limits, and to identify which resource is likely
to be limiting at the optimum.

All numbers below refer to the submission benchmark used by
`tests/submission_tests.py`: `forest_height=10`, `rounds=16`, `batch_size=256`.

---

## 1) What the kernel does

For each round `h` and batch element `i`:

- Load `idx = mem[inp_indices_p + i]`
- Load `val = mem[inp_values_p + i]`
- Load `node_val = mem[forest_values_p + idx]`
- Compute `val = myhash(val ^ node_val)` where `myhash` has 6 stages
- Update `idx = 2*idx + (1 if val even else 2)` and wrap `idx = 0 if idx >= n_nodes`
- Store back `val` and `idx`

This is a "batch of walkers" through an implicit binary tree (heap indexing).

---

## 2) Machine constraints that matter

One "instruction bundle" can contain multiple "slots" per engine; the machine
executes all engines for the bundle in one cycle. Slot limits per cycle:

- `valu`: 6 vector-ALU slots/cycle
- `alu`: 12 scalar-ALU slots/cycle
- `load`: 2 load slots/cycle (scalar load/const or vload)
- `store`: 2 store slots/cycle (scalar store or vstore)
- `flow`: 1 flow slot/cycle (select/jumps/etc)

Vector length `VLEN = 8`.

Important semantic constraint: effects of instructions become visible only at
end-of-cycle (writes are deferred), so you cannot chain dependent ops within
the same cycle even if slot capacity exists.

---

## 3) Work sizing

We have `rounds * batch_size = 16 * 256 = 4096` element-updates.

In vectors, that's `4096 / VLEN = 512` "vector groups" (each group processes 8
elements).

---

## 4) The dominant theoretical lower bound: hash throughput (valu engine)

The per-element heavy compute is `myhash`. In `problem.py`, each stage looks like:

Stage template:
`a = r( op2( r(op1(a, val1)), r(op3(a, val3)) ) )`

with `(op1, op2, op3)` drawn from `{+, ^}` and `op3` being `<<` or `>>`.

### 4.1) Best-case valu-slot count per vector group

Using the available vector ops:

- We must do the initial `val ^ node_val`: 1 valu slot.
- Stages 1, 3, 5 have `op1="+"`, `op2="+"`, and `op3="<<"`.
  These can be collapsed with `("multiply_add", ...)` because:
  `(a + c) + (a << s) = a*(1 + 2^s) + c`.
  So each of those stages can be 1 valu slot (plus preloaded constants).
- Stages 2 and 6 have `op1="^"`, `op2="^"`, `op3=">>"`.
  These fundamentally require:
  1) shift, 2) xor with val1, 3) xor with shifted term => 3 valu slots/stage.
- Stage 4 has `op1="+"`, `op2="+"`, `op3="<<"` but with `op2="^"` in the table:
  `("+", 0xD3A2646C, "^", "<<", 9)`.
  That requires:
  1) add, 2) shift, 3) xor => 3 valu slots.

Thus a minimal valu-op count per vector group is:

- 1 (initial xor)
- + 1 + 3 + 1 + 3 + 1 + 3 (six stages)
= 13 valu slots / vector group

### 4.2) Convert valu-slots to cycles

Total valu slots >= `512 * 13 = 6656`.

With `SLOT_LIMITS["valu"] = 6`, a hard lower bound is:

`ceil(6656 / 6) = 1110` cycles

This bound is independent of memory traffic and scalar bookkeeping. It is a
"physics" limit: you cannot go below ~1110 cycles without changing the hash or
the valu slot limit.

---

## 5) Memory rooflines (load/store engines)

### 5.1) Naive gather-like node loads would dominate

The access `forest_values_p + idx` is a per-element indexed load (a gather).
This ISA does not have vgather; naively you do 1 scalar load per element:

- Node loads: 4096 scalar loads
- Load capacity: 2 loads/cycle
- Lower bound: `ceil(4096 / 2) = 2048` cycles

If you pay this full cost, you can never reach the 1400s; so any good solution
must avoid "4096 independent node loads".

### 5.2) Best-case reuse bound on node loads (specific to this workload)

All indices start at 0. On a perfect binary tree, after `d` steps from root,
there are at most `2^d` distinct nodes reachable. With a batch of 256 walkers,
the number of *distinct* indices at depth `d` is at most `min(2^d, 256)`.

Within one 16-step round loop, walkers descend 10 levels to leaves, then wrap
to root and repeat. So depths by step are:

- steps 0..9: depth 0..9
- steps 10..15: depth 0..5 (after wrap)

Best-case distinct-node loads per round-step:

`sum_{d=0..9} 2^d + sum_{d=0..5} 2^d
 = (2^10 - 1) + (2^6 - 1)
 = 1023 + 63
 = 1086`

But once `2^d` exceeds batch size, it's capped at 256; in this benchmark it
does not exceed 256 until depth 8, so a tighter tally for 16 steps is:

- d=0..7: `1+2+4+8+16+32+64+128 = 255`
- d=8..9: `256+256 = 512`
- d=0..5 (after wrap): `1+2+4+8+16+32 = 63`

Total best-case distinct node loads across all 16 steps:

`255 + 512 + 63 = 830` scalar node loads

Load-cycles lower bound for node loads in this ideal reuse scenario:

`ceil(830 / 2) = 415` cycles

This is *below* the 1110 valu bound, meaning that at the theoretical optimum,
node loads could be made non-limiting by aggressively exploiting reuse.

Reality will be worse than this best case because the branch decision depends
on `val`, which depends on the hash; so you do not get to choose a perfectly
balanced reuse pattern. But it shows why "beating 2000 cycles" is plausible:
there is enough structure (shared ancestry + wrap) for reuse to exist.

### 5.3) Input loads/stores are not the bottleneck

Per element update, you must at least read and write back:

- loads: `idx` + `val` => 2 loads/element => 8192 loads
- stores: `idx` + `val` => 2 stores/element => 8192 stores

Those sound huge, but they can be vectorized (contiguous):

- Use `vload` / `vstore` for 8-wide blocks.
- Per vector group: 2 vloads + 2 vstores.
- Total vloads: `512 * 2 = 1024` => `ceil(1024 / 2) = 512` load cycles
- Total vstores: `512 * 2 = 1024` => `ceil(1024 / 2) = 512` store cycles

These are also below 1110, so they should not be limiting if scheduled well.

---

## 6) Scalar/flow overhead lower bounds (usually secondary)

Even with vectorized compute, you still need:

- Some scalar/flow work to generate addresses, handle wrap/selection, organize
  reuse strategy, etc.
- But `alu` has 12 slots/cycle and `flow` 1 slot/cycle, typically making them
  less constraining than `valu` unless you introduce heavy branching or serialize.

The absolute minimum cycle count therefore tends to be set by the `valu` bound,
with additional overhead pushing the achievable optimum above 1110.

---

## 7) Bottom line: what is the absolute minimum?

- Hard theoretical floor from valu throughput: ~1110 cycles for this benchmark.
- Practical optimum must be higher due to:
  - dependency chains across stages (end-of-cycle visibility)
  - nonzero scalar/flow bookkeeping
  - imperfect node-load reuse (no true gather; reuse strategy has constraints)

My expectation for a "real" best possible solution (without changing the rules)
is somewhere above 1110, likely in the 1150-1250 range.

Anything below ~1110 should be impossible without changing the hash/ISA or
exploiting a bug in the simulator/tests.

