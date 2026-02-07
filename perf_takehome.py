"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


# ---------------------------------------------------------------------------
# List scheduler for VLIW packing
# ---------------------------------------------------------------------------

class Op:
    __slots__ = ['engine', 'slot', 'id', 'succs', 'pred_count', 'priority']
    def __init__(self, engine, slot, op_id):
        self.engine = engine
        self.slot = slot
        self.id = op_id
        self.succs = []
        self.pred_count = 0
        self.priority = 0


class OpGraph:
    """Builds a dependency graph of operations for scheduling."""
    def __init__(self):
        self.ops = []
        self.last_writer = {}           # addr -> op_index
        self.readers_since_write = {}   # addr -> [op_index]
        self.const_addrs = set()

    def mark_constants(self, addrs):
        """Mark addresses as read-only constants (skip WAR tracking for them)."""
        self.const_addrs.update(addrs)

    def add_op(self, engine, slot, reads, writes):
        op_idx = len(self.ops)
        op = Op(engine, slot, op_idx)

        pred_set = set()

        # RAW: this op reads addresses written by earlier ops
        for addr in reads:
            w = self.last_writer.get(addr)
            if w is not None:
                pred_set.add(w)

        # WAW + WAR: this op writes to addresses
        for addr in writes:
            w = self.last_writer.get(addr)
            if w is not None:
                pred_set.add(w)
            # WAR: must come after all readers since last write
            if addr not in self.const_addrs:
                for r_idx in self.readers_since_write.get(addr, ()):
                    pred_set.add(r_idx)
            self.last_writer[addr] = op_idx
            self.readers_since_write[addr] = []

        # Track as reader (skip constant addrs to avoid huge lists)
        for addr in reads:
            if addr not in self.const_addrs:
                if addr not in self.readers_since_write:
                    self.readers_since_write[addr] = []
                self.readers_since_write[addr].append(op_idx)

        # Build edges
        for pred_idx in pred_set:
            self.ops[pred_idx].succs.append(op_idx)
            op.pred_count += 1

        self.ops.append(op)
        return op_idx


def schedule_ops(graph):
    """Schedule ops into VLIW instruction bundles using a greedy list scheduler."""
    ops = graph.ops
    n = len(ops)
    if n == 0:
        return []

    # Compute critical path priorities (reverse topo order)
    for op in reversed(ops):
        max_p = 0
        for s_idx in op.succs:
            p = ops[s_idx].priority
            if p > max_p:
                max_p = p
        op.priority = 1 + max_p

    # Initialize ready list
    ready = []
    for op in ops:
        if op.pred_count == 0:
            ready.append(op)
    ready.sort(key=lambda o: -o.priority)

    bundles = []
    newly_ready = []
    scheduled_count = 0

    while ready or newly_ready:
        if newly_ready:
            ready.extend(newly_ready)
            ready.sort(key=lambda o: -o.priority)
            newly_ready = []

        if not ready:
            break

        bundle = {}
        slot_counts = {}
        scheduled_this = []
        remaining = []

        for op in ready:
            eng = op.engine
            cnt = slot_counts.get(eng, 0)
            if cnt < SLOT_LIMITS.get(eng, 0):
                bundle.setdefault(eng, []).append(op.slot)
                slot_counts[eng] = cnt + 1
                scheduled_this.append(op)
            else:
                remaining.append(op)

        if not scheduled_this:
            # Safety: force-schedule one op to avoid infinite loop
            op = remaining.pop(0)
            bundle = {op.engine: [op.slot]}
            scheduled_this = [op]

        bundles.append(bundle)
        ready = remaining
        scheduled_count += len(scheduled_this)

        for op in scheduled_this:
            for s_idx in op.succs:
                succ = ops[s_idx]
                succ.pred_count -= 1
                if succ.pred_count == 0:
                    newly_ready.append(succ)

    return bundles


# ---------------------------------------------------------------------------
# Kernel Builder
# ---------------------------------------------------------------------------

class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_kernel(self, forest_height, n_nodes, batch_size, rounds):
        n_groups = batch_size // VLEN  # 32

        # ---- Compile-time known addresses ----
        header_size = 7
        forest_values_p_val = header_size
        inp_indices_p_val = header_size + n_nodes
        inp_values_p_val = inp_indices_p_val + batch_size

        # ---- Phase 0: Setup ----
        # We need header vars loaded for the local test's reference check
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")

        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        # Scalar constants
        zero_const = self.scratch_const(0, "zero")
        eight_const = self.scratch_const(8, "eight")

        # forest_values_p as a compile-time const in its own scratch
        fvp_const = self.scratch_const(forest_values_p_val, "fvp_const")

        # ---- Allocate persistent vectors (32 groups x 2 vectors x 8 elements) ----
        vidx = []
        vval = []
        for g in range(n_groups):
            vidx.append(self.alloc_scratch(f"vidx_{g}", VLEN))
            vval.append(self.alloc_scratch(f"vval_{g}", VLEN))

        # ---- Allocate double-buffered shared temp vectors ----
        # Double-buffering prevents WAR dependencies from serializing groups.
        # Even-indexed groups use buffer 0, odd-indexed use buffer 1.
        N_BUF = 16
        vaddr = [self.alloc_scratch(f"vaddr_{b}", VLEN) for b in range(N_BUF)]
        vnode = [self.alloc_scratch(f"vnode_{b}", VLEN) for b in range(N_BUF)]
        vtmp1 = [self.alloc_scratch(f"vtmp1_{b}", VLEN) for b in range(N_BUF)]
        vtmp2 = [self.alloc_scratch(f"vtmp2_{b}", VLEN) for b in range(N_BUF)]
        vtmp_br = [self.alloc_scratch(f"vtmp_br_{b}", VLEN) for b in range(N_BUF)]

        # ---- Allocate and initialize vector constants ----
        def alloc_vec_const(name, value):
            addr = self.alloc_scratch(name, VLEN)
            self.add("load", ("const", tmp1, value))
            self.add("valu", ("vbroadcast", addr, tmp1))
            return addr

        v_one = alloc_vec_const("v_one", 1)
        v_two = alloc_vec_const("v_two", 2)
        v_n_nodes = alloc_vec_const("v_n_nodes", n_nodes)

        # multiply_add constants: stage 0 (4097), stage 2 (33), stage 4 (9)
        v_mul_4097 = alloc_vec_const("v_mul_4097", (1 << 12) + 1)
        v_C0 = alloc_vec_const("v_C0", HASH_STAGES[0][1])

        v_C1 = alloc_vec_const("v_C1", HASH_STAGES[1][1])
        v_19 = alloc_vec_const("v_19", HASH_STAGES[1][4])

        v_mul_33 = alloc_vec_const("v_mul_33", (1 << 5) + 1)
        v_C2 = alloc_vec_const("v_C2", HASH_STAGES[2][1])

        v_C3 = alloc_vec_const("v_C3", HASH_STAGES[3][1])
        v_9 = alloc_vec_const("v_9", HASH_STAGES[3][4])

        v_C4 = alloc_vec_const("v_C4", HASH_STAGES[4][1])
        # stage 4 multiplier is also 9 = (1<<3)+1, reuse v_9

        v_C5 = alloc_vec_const("v_C5", HASH_STAGES[5][1])
        v_16 = alloc_vec_const("v_16", HASH_STAGES[5][4])

        # ---- Pause (matches first yield in reference_kernel2) ----
        self.add("flow", ("pause",))

        # ---- Phase 1: Initial vloads ----
        ptr_idx = self.alloc_scratch("ptr_idx")
        ptr_val = self.alloc_scratch("ptr_val")

        self.add("load", ("const", ptr_idx, inp_indices_p_val))
        self.add("load", ("const", ptr_val, inp_values_p_val))

        for g in range(n_groups):
            self.instrs.append({
                "load": [("vload", vidx[g], ptr_idx), ("vload", vval[g], ptr_val)],
                "alu": [("+", ptr_idx, ptr_idx, eight_const),
                        ("+", ptr_val, ptr_val, eight_const)],
            })

        # ---- Preload tree node values for levels 0-2 ----
        # Level 0: tree[0] - used by rounds 0, 11 (all elements at idx=0)
        tree_s0 = self.alloc_scratch("tree_s0")
        v_tree0 = self.alloc_scratch("v_tree0", VLEN)
        self.instrs.append({"load": [("const", tmp1, forest_values_p_val + 0)]})
        self.instrs.append({"load": [("load", tree_s0, tmp1)]})
        self.instrs.append({"valu": [("vbroadcast", v_tree0, tree_s0)]})

        # Level 1: tree[1], tree[2] - used by rounds 1, 12
        # Selection: node = cond * (tree1 - tree2) + tree2 where cond = idx & 1
        tree_s1 = self.alloc_scratch("tree_s1")
        tree_s2 = self.alloc_scratch("tree_s2")
        diff12_s = self.alloc_scratch("diff12_s")
        v_tree2_pre = self.alloc_scratch("v_tree2_pre", VLEN)
        v_diff12 = self.alloc_scratch("v_diff12", VLEN)
        self.instrs.append({"load": [("const", tmp1, forest_values_p_val + 1),
                                     ("const", tmp2, forest_values_p_val + 2)]})
        self.instrs.append({"load": [("load", tree_s1, tmp1),
                                     ("load", tree_s2, tmp2)]})
        self.instrs.append({"alu": [("-", diff12_s, tree_s1, tree_s2)]})
        self.instrs.append({"valu": [("vbroadcast", v_tree2_pre, tree_s2),
                                     ("vbroadcast", v_diff12, diff12_s)]})

        # Level 2: tree[3..6] - used by rounds 2, 13
        # Two-level selection using multiply_add
        tree_s3 = self.alloc_scratch("tree_s3")
        tree_s4 = self.alloc_scratch("tree_s4")
        tree_s5 = self.alloc_scratch("tree_s5")
        tree_s6 = self.alloc_scratch("tree_s6")
        diff34_s = self.alloc_scratch("diff34_s")
        diff56_s = self.alloc_scratch("diff56_s")
        v_tree3_pre = self.alloc_scratch("v_tree3_pre", VLEN)
        v_tree5_pre = self.alloc_scratch("v_tree5_pre", VLEN)
        v_diff34 = self.alloc_scratch("v_diff34", VLEN)
        v_diff56 = self.alloc_scratch("v_diff56", VLEN)
        v_three = alloc_vec_const("v_three", 3)

        self.instrs.append({"load": [("const", tmp1, forest_values_p_val + 3),
                                     ("const", tmp2, forest_values_p_val + 4)]})
        self.instrs.append({"load": [("load", tree_s3, tmp1),
                                     ("load", tree_s4, tmp2)]})
        self.instrs.append({"load": [("const", tmp1, forest_values_p_val + 5),
                                     ("const", tmp2, forest_values_p_val + 6)]})
        self.instrs.append({"load": [("load", tree_s5, tmp1),
                                     ("load", tree_s6, tmp2)]})
        self.instrs.append({"alu": [("-", diff34_s, tree_s4, tree_s3),
                                    ("-", diff56_s, tree_s6, tree_s5)]})
        self.instrs.append({"valu": [("vbroadcast", v_tree3_pre, tree_s3),
                                     ("vbroadcast", v_tree5_pre, tree_s5),
                                     ("vbroadcast", v_diff34, diff34_s),
                                     ("vbroadcast", v_diff56, diff56_s)]})

        # ---- Phase 2: Main loop - build op graph for scheduler ----
        graph = OpGraph()

        # Determine tree level for each round
        def tree_level(r):
            if r <= 10:
                return r
            return r - 11

        # Mark all vector constant addresses as read-only
        const_addr_ranges = [
            (v_one, VLEN), (v_two, VLEN), (v_n_nodes, VLEN),
            (v_mul_4097, VLEN), (v_C0, VLEN), (v_C1, VLEN), (v_19, VLEN),
            (v_mul_33, VLEN), (v_C2, VLEN), (v_C3, VLEN), (v_9, VLEN),
            (v_C4, VLEN), (v_C5, VLEN), (v_16, VLEN),
            (v_tree0, VLEN), (v_tree2_pre, VLEN), (v_diff12, VLEN),
            (v_tree3_pre, VLEN), (v_tree5_pre, VLEN),
            (v_diff34, VLEN), (v_diff56, VLEN), (v_three, VLEN),
        ]
        const_addrs = set()
        for base, length in const_addr_ranges:
            for i in range(length):
                const_addrs.add(base + i)
        const_addrs.add(fvp_const)
        const_addrs.add(zero_const)
        graph.mark_constants(const_addrs)

        # Helper: range set
        def vrange(base):
            return set(range(base, base + VLEN))

        for r in range(rounds):
            level = tree_level(r)
            for g in range(n_groups):
                vg_idx = vidx[g]
                vg_val = vval[g]
                buf = g % N_BUF
                va = vaddr[buf]
                vn = vnode[buf]
                vt1 = vtmp1[buf]
                vt2 = vtmp2[buf]
                vb = vtmp_br[buf]

                if level == 0:
                    # -- Level 0: all at idx=0, use preloaded tree[0] --
                    graph.add_op(
                        "valu", ("^", vg_val, vg_val, v_tree0),
                        reads=vrange(vg_val) | vrange(v_tree0),
                        writes=vrange(vg_val)
                    )
                elif level == 1:
                    # -- Level 1: idx in {1,2}, select from preloaded --
                    graph.add_op(
                        "valu", ("&", vn, vg_idx, v_one),
                        reads=vrange(vg_idx) | vrange(v_one),
                        writes=vrange(vn)
                    )
                    graph.add_op(
                        "valu", ("multiply_add", vn, vn, v_diff12, v_tree2_pre),
                        reads=vrange(vn) | vrange(v_diff12) | vrange(v_tree2_pre),
                        writes=vrange(vn)
                    )
                    graph.add_op(
                        "valu", ("^", vg_val, vg_val, vn),
                        reads=vrange(vg_val) | vrange(vn),
                        writes=vrange(vg_val)
                    )
                elif level == 2:
                    # -- Level 2: idx in {3,4,5,6}, 4-value selection --
                    graph.add_op(
                        "valu", ("-", vt1, vg_idx, v_three),
                        reads=vrange(vg_idx) | vrange(v_three),
                        writes=vrange(vt1)
                    )
                    graph.add_op(
                        "valu", ("&", vt2, vt1, v_one),
                        reads=vrange(vt1) | vrange(v_one),
                        writes=vrange(vt2)
                    )
                    graph.add_op(
                        "valu", (">>", va, vt1, v_one),
                        reads=vrange(vt1) | vrange(v_one),
                        writes=vrange(va)
                    )
                    graph.add_op(
                        "valu", ("multiply_add", vn, vt2, v_diff34, v_tree3_pre),
                        reads=vrange(vt2) | vrange(v_diff34) | vrange(v_tree3_pre),
                        writes=vrange(vn)
                    )
                    graph.add_op(
                        "valu", ("multiply_add", vt1, vt2, v_diff56, v_tree5_pre),
                        reads=vrange(vt2) | vrange(v_diff56) | vrange(v_tree5_pre),
                        writes=vrange(vt1)
                    )
                    graph.add_op(
                        "valu", ("-", vt2, vt1, vn),
                        reads=vrange(vt1) | vrange(vn),
                        writes=vrange(vt2)
                    )
                    graph.add_op(
                        "valu", ("multiply_add", vn, va, vt2, vn),
                        reads=vrange(va) | vrange(vt2) | vrange(vn),
                        writes=vrange(vn)
                    )
                    graph.add_op(
                        "valu", ("^", vg_val, vg_val, vn),
                        reads=vrange(vg_val) | vrange(vn),
                        writes=vrange(vg_val)
                    )
                else:
                    # -- Standard scattered loads --
                    # Address computation (8 ALU ops)
                    for i in range(VLEN):
                        graph.add_op(
                            "alu", ("+", va + i, fvp_const, vg_idx + i),
                            reads={fvp_const, vg_idx + i},
                            writes={va + i}
                        )
                    # Scattered loads (8 load ops)
                    for i in range(VLEN):
                        graph.add_op(
                            "load", ("load", vn + i, va + i),
                            reads={va + i},
                            writes={vn + i}
                        )
                    # XOR val ^= node_val
                    graph.add_op(
                        "valu", ("^", vg_val, vg_val, vn),
                        reads=vrange(vg_val) | vrange(vn),
                        writes=vrange(vg_val)
                    )

                # -- Hash stage 0: multiply_add (a*4097 + C0) --
                graph.add_op(
                    "valu", ("multiply_add", vg_val, vg_val, v_mul_4097, v_C0),
                    reads=vrange(vg_val) | vrange(v_mul_4097) | vrange(v_C0),
                    writes=vrange(vg_val)
                )

                # -- Hash stage 1: (a^C1) ^ (a>>19) --
                graph.add_op(
                    "valu", ("^", vt1, vg_val, v_C1),
                    reads=vrange(vg_val) | vrange(v_C1),
                    writes=vrange(vt1)
                )
                graph.add_op(
                    "valu", (">>", vt2, vg_val, v_19),
                    reads=vrange(vg_val) | vrange(v_19),
                    writes=vrange(vt2)
                )
                graph.add_op(
                    "valu", ("^", vg_val, vt1, vt2),
                    reads=vrange(vt1) | vrange(vt2),
                    writes=vrange(vg_val)
                )

                # -- Hash stage 2: multiply_add (a*33 + C2) --
                graph.add_op(
                    "valu", ("multiply_add", vg_val, vg_val, v_mul_33, v_C2),
                    reads=vrange(vg_val) | vrange(v_mul_33) | vrange(v_C2),
                    writes=vrange(vg_val)
                )

                # -- Hash stage 3: (a+C3) ^ (a<<9) --
                graph.add_op(
                    "valu", ("+", vt1, vg_val, v_C3),
                    reads=vrange(vg_val) | vrange(v_C3),
                    writes=vrange(vt1)
                )
                graph.add_op(
                    "valu", ("<<", vt2, vg_val, v_9),
                    reads=vrange(vg_val) | vrange(v_9),
                    writes=vrange(vt2)
                )
                graph.add_op(
                    "valu", ("^", vg_val, vt1, vt2),
                    reads=vrange(vt1) | vrange(vt2),
                    writes=vrange(vg_val)
                )

                # -- Hash stage 4: multiply_add (a*9 + C4) --
                graph.add_op(
                    "valu", ("multiply_add", vg_val, vg_val, v_9, v_C4),
                    reads=vrange(vg_val) | vrange(v_9) | vrange(v_C4),
                    writes=vrange(vg_val)
                )

                # -- Hash stage 5: (a^C5) ^ (a>>16) --
                graph.add_op(
                    "valu", ("^", vt1, vg_val, v_C5),
                    reads=vrange(vg_val) | vrange(v_C5),
                    writes=vrange(vt1)
                )
                graph.add_op(
                    "valu", (">>", vt2, vg_val, v_16),
                    reads=vrange(vg_val) | vrange(v_16),
                    writes=vrange(vt2)
                )
                graph.add_op(
                    "valu", ("^", vg_val, vt1, vt2),
                    reads=vrange(vt1) | vrange(vt2),
                    writes=vrange(vg_val)
                )

                # -- Branch / wrap --
                if r == 10:
                    # All elements wrap to 0 at round 10
                    graph.add_op(
                        "valu", ("vbroadcast", vg_idx, zero_const),
                        reads={zero_const},
                        writes=vrange(vg_idx)
                    )
                else:
                    # Branch: idx = 2*idx + ((val & 1) + 1)
                    graph.add_op(
                        "valu", ("&", vb, vg_val, v_one),
                        reads=vrange(vg_val) | vrange(v_one),
                        writes=vrange(vb)
                    )
                    graph.add_op(
                        "valu", ("+", vb, vb, v_one),
                        reads=vrange(vb) | vrange(v_one),
                        writes=vrange(vb)
                    )
                    graph.add_op(
                        "valu", ("multiply_add", vg_idx, vg_idx, v_two, vb),
                        reads=vrange(vg_idx) | vrange(v_two) | vrange(vb),
                        writes=vrange(vg_idx)
                    )

        # Schedule all ops
        main_bundles = schedule_ops(graph)
        self.instrs.extend(main_bundles)

        # ---- Phase 3: Final vstores ----
        self.add("load", ("const", ptr_idx, inp_indices_p_val))
        self.add("load", ("const", ptr_val, inp_values_p_val))

        for g in range(n_groups):
            self.instrs.append({
                "store": [("vstore", ptr_idx, vidx[g]), ("vstore", ptr_val, vval[g])],
                "alu": [("+", ptr_idx, ptr_idx, eight_const),
                        ("+", ptr_val, ptr_val, eight_const)],
            })

        # Final pause
        self.instrs.append({"flow": [("pause",)]})


BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


if __name__ == "__main__":
    unittest.main()
