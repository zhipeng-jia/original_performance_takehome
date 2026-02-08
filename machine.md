# VLIW SIMD Machine — ISA Reference

## 1. Architecture Overview

A single-core **Very Long Instruction Word (VLIW)** processor with **SIMD** vector support. The processor executes one "instruction bundle" per cycle. Each bundle can contain operations for multiple execution engines simultaneously. All engines read their inputs at the start of the cycle and commit writes at the end — giving **write-after-read (WAR) semantics within a cycle**.

There is **no out-of-order execution**, no branch predictor, and no hardware hazard detection. All scheduling is the compiler's responsibility.

### Key Constants

| Parameter | Value | Notes |
|-----------|-------|-------|
| `VLEN` | 8 | Elements per SIMD vector |
| `N_CORES` | 1 | Single-core in this version |
| `SCRATCH_SIZE` | 1536 | Words of scratch memory (serves as register file) |

### Execution Engines and Slot Limits

Each engine can execute up to its slot limit per cycle. All engines fire in parallel.

| Engine | Slots/Cycle | Purpose |
|--------|-------------|---------|
| **ALU** | 12 | Scalar 32-bit arithmetic and bitwise ops |
| **VALU** | 6 | Vector (VLEN=8) arithmetic, bitwise, broadcast, multiply-add |
| **Load** | 2 | Scalar/vector loads from memory, constant materialization |
| **Store** | 2 | Scalar/vector stores to memory |
| **Flow** | 1 | Control flow, select, jumps, pause/halt |
| *Debug* | 64 | Debug assertions (0-cycle, not counted) |

**Maximum throughput per cycle:** 12 ALU + 6 VALU + 2 Load + 2 Store + 1 Flow = 23 operations.

## 2. Memory Model

### Memory (Main Memory)
- Flat array of 32-bit words, addressed by integer index.
- Accessed via Load (read) and Store (write) engines.
- Persists across `run()` invocations (between `pause` instructions).

### Scratch Space
- **1536-word** array of 32-bit words, private to each core.
- Serves as: **registers**, **vector registers**, **constant pool**, **manually-managed cache**.
- All ALU/VALU operands are scratch addresses (not memory addresses).
- There is **no indirect scratch access** — all operand addresses are determined at code-generation time.
- Initialized to all zeros.

### Write Semantics
- **Within a cycle**: All reads happen first, then all writes commit atomically at cycle end.
- This means two operations in the same bundle can read and write the same scratch address: the reader sees the **old** value, the writer's new value takes effect after the cycle.
- If two operations in the same bundle write to the same scratch address, the last one processed wins (engine processing order: alu, valu, load, store, flow).

## 3. Instruction Bundle Format

Each instruction is a Python `dict` mapping engine names to lists of "slots":

```python
{
    "valu": [("*", 4, 0, 0), ("+", 8, 4, 0)],
    "load": [("load", 16, 17)],
    "alu":  [("+", 100, 101, 102)]
}
```

**Every number in a slot is a scratch address**, except:
- `const` immediate values
- Jump/branch target addresses (PC values)
- The first operand of most instructions is the **destination** (exceptions: `store`, some flow ops)

---

## 4. ALU Engine — Scalar Operations

**12 slots per cycle.** Each slot performs one scalar 32-bit operation.

### Format: `(op, dest, a1, a2)`

All operands are scratch addresses. The operation reads `scratch[a1]` and `scratch[a2]`, computes the result, and writes to `scratch[dest]`.

### Operations

| Op | Semantics | Notes |
|----|-----------|-------|
| `"+"` | `a1 + a2` | Unsigned 32-bit wrap |
| `"-"` | `a1 - a2` | Unsigned 32-bit wrap |
| `"*"` | `a1 * a2` | Unsigned 32-bit wrap |
| `"//"` | `a1 // a2` | Integer floor division |
| `"cdiv"` | `ceil(a1 / a2)` | Ceiling division |
| `"^"` | `a1 ^ a2` | Bitwise XOR |
| `"&"` | `a1 & a2` | Bitwise AND |
| `"\|"` | `a1 \| a2` | Bitwise OR |
| `"<<"` | `a1 << a2` | Left shift |
| `">>"` | `a1 >> a2` | Right shift (logical) |
| `"%"` | `a1 % a2` | Modulo |
| `"<"` | `int(a1 < a2)` | Returns 0 or 1 |
| `"=="` | `int(a1 == a2)` | Returns 0 or 1 |

All results are taken modulo `2^32`.

---

## 5. VALU Engine — Vector Operations

**6 slots per cycle.** Each slot operates on VLEN=8 element vectors.

Vector operands use **contiguous scratch addresses**: a vector at base address `b` occupies `scratch[b], scratch[b+1], ..., scratch[b+7]`.

### Standard vector ops: `(op, dest, a1, a2)`

Same operation set as ALU, but applied element-wise across VLEN=8 elements:
```
for i in range(8):
    scratch[dest+i] = op(scratch[a1+i], scratch[a2+i]) % 2^32
```

Supports all ALU operations: `+`, `-`, `*`, `//`, `^`, `&`, `|`, `<<`, `>>`, `%`, `<`, `==`, etc.

### Special VALU operations

#### `vbroadcast`: `("vbroadcast", dest, src)`
Broadcasts a **scalar** scratch value to all VLEN elements:
```
for i in range(8):
    scratch[dest+i] = scratch[src]
```
- `src` is a **scalar** address (single word)
- `dest` is a **vector** base address (8 words)

#### `multiply_add`: `("multiply_add", dest, a, b, c)`
Fused multiply-add, element-wise:
```
for i in range(8):
    scratch[dest+i] = (scratch[a+i] * scratch[b+i] + scratch[c+i]) % 2^32
```
- All four operands are vector base addresses (8 elements each)
- Single slot, single cycle — **extremely powerful** for collapsing affine transforms

---

## 6. Load Engine

**2 slots per cycle.** Reads data from main memory into scratch, or materializes constants.

### `load`: `("load", dest, addr)`
Reads one word from main memory:
```
scratch[dest] = mem[scratch[addr]]
```
- `dest`: scratch address for result (scalar)
- `addr`: scratch address containing the memory address to read from

### `load_offset`: `("load_offset", dest, addr, offset)`
Like `load` but with an offset applied to both scratch addresses:
```
scratch[dest + offset] = mem[scratch[addr + offset]]
```
- Useful for loading individual elements of a vector

### `vload`: `("vload", dest, addr)`
Vector load of 8 contiguous words from memory:
```
base_addr = scratch[addr]  # addr is a SCALAR scratch address
for i in range(8):
    scratch[dest+i] = mem[base_addr + i]
```
- `addr` is a **scalar** scratch address holding the base memory address
- `dest` is a **vector** base address (8 words written)
- Only works for **contiguous** memory — no scatter/gather support

### `const`: `("const", dest, val)`
Materialize an immediate constant into scratch:
```
scratch[dest] = val % 2^32
```
- `val` is an immediate integer value (not a scratch address!)
- `dest` is a scalar scratch address

---

## 7. Store Engine

**2 slots per cycle.** Writes data from scratch to main memory.

### `store`: `("store", addr, src)`
Write one word to main memory:
```
mem[scratch[addr]] = scratch[src]
```
- Note: `addr` comes first (unlike load where dest comes first)

### `vstore`: `("vstore", addr, src)`
Vector store of 8 contiguous words to memory:
```
base_addr = scratch[addr]  # addr is a SCALAR scratch address
for i in range(8):
    mem[base_addr + i] = scratch[src + i]
```

---

## 8. Flow Engine

**1 slot per cycle.** Control flow and conditional operations.

### `select`: `("select", dest, cond, a, b)`
Scalar conditional select:
```
scratch[dest] = scratch[a] if scratch[cond] != 0 else scratch[b]
```

### `vselect`: `("vselect", dest, cond, a, b)`
Vector conditional select (element-wise):
```
for i in range(8):
    scratch[dest+i] = scratch[a+i] if scratch[cond+i] != 0 else scratch[b+i]
```

### `add_imm`: `("add_imm", dest, a, imm)`
Add immediate (scalar):
```
scratch[dest] = (scratch[a] + imm) % 2^32
```
- `imm` is an immediate integer, not a scratch address

### `halt`: `("halt",)`
Stops the core permanently.

### `pause`: `("pause",)`
Pauses the core. Execution resumes on the next `machine.run()` call. Used for synchronization with reference kernel yields.

### `trace_write`: `("trace_write", val)`
Appends `scratch[val]` to the core's trace buffer.

### `cond_jump`: `("cond_jump", cond, addr)`
Conditional jump (absolute):
```
if scratch[cond] != 0: PC = addr
```

### `cond_jump_rel`: `("cond_jump_rel", cond, offset)`
Conditional jump (relative):
```
if scratch[cond] != 0: PC += offset
```

### `jump`: `("jump", addr)`
Unconditional jump (absolute): `PC = addr`

### `jump_indirect`: `("jump_indirect", addr)`
Unconditional indirect jump: `PC = scratch[addr]`

### `coreid`: `("coreid", dest)`
Writes core ID to scratch: `scratch[dest] = core.id`

---

## 9. Debug Engine

**64 slots per cycle.** Zero-cost assertions (do not count as a cycle).

### `compare`: `("compare", loc, key)`
Assert that `scratch[loc]` equals `value_trace[key]`.

### `vcompare`: `("vcompare", loc, keys)`
Assert that `scratch[loc:loc+8]` equals `[value_trace[k] for k in keys]`.

Debug instructions are only executed when `enable_debug` is True and do not contribute to cycle count (a bundle with only debug ops does not count as a cycle).

---

## 10. Cycle Counting Rules

1. Each instruction bundle = 1 cycle (if it contains at least one non-debug engine).
2. Bundles with only `debug` slots are **free** (0 cycles).
3. The `cycle` counter increments after processing all cores for a given cycle.
4. A `pause` instruction counts as a cycle.
5. PC advances by 1 per cycle (sequentially through the instruction list), unless modified by a jump.

---

## 11. Programming Model Notes

### Scratch as Registers
Since all operands are scratch addresses, scratch space functions as a massive register file. You allocate "registers" by claiming scratch addresses at compile time.

### No Register Renaming
The processor has no hardware register renaming. All hazard management (RAW, WAR, WAW) must be handled by the compiler/scheduler.

### Write-After-Read Safety
Within a single bundle, reads happen before writes. This means:
- Two ops in the same bundle can safely read from an address that another op in the same bundle writes to — they see the old value.
- This enables certain "swap" patterns without temporaries.

### Vector Memory Access Patterns
- **Contiguous**: `vload`/`vstore` — 1 load/store slot, accesses 8 contiguous memory words
- **Scattered**: Must use 8 individual `load` ops — costs 8 load slots (4 cycles minimum at 2 loads/cycle)
- **Broadcast**: Use `const` (1 load slot) + `vbroadcast` (1 VALU slot) to broadcast a scalar to all 8 vector lanes

### Resource Pressure Summary
For a typical inner loop iteration processing 8 elements:

| Resource | Capacity | Typical Demand | Bottleneck Risk |
|----------|----------|---------------|-----------------|
| VALU (6/cycle) | 6 | 12-20 ops/group | **HIGH** — often the bottleneck |
| Load (2/cycle) | 2 | 0-8 ops/group | **HIGH** for scattered access |
| ALU (12/cycle) | 12 | 8-24 ops/group | Medium — usually not limiting |
| Store (2/cycle) | 2 | 0 (final only) | Low |
| Flow (1/cycle) | 1 | 0 (eliminated) | Low |
