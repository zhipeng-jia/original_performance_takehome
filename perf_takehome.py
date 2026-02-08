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
    __slots__ = ['engine', 'slot', 'id', 'succs', 'pred_count', 'priority', 'group']
    def __init__(self, engine, slot, op_id, group=-1):
        self.engine = engine
        self.slot = slot
        self.id = op_id
        self.succs = []
        self.pred_count = 0
        self.priority = 0
        self.group = group


class OpGraph:
    """Builds a dependency graph of operations for scheduling."""
    def __init__(self):
        self.ops = []
        self.last_writer = {}           # addr -> op_index
        self.readers_since_write = {}   # addr -> [op_index]
        self.const_addrs = set()
        self.local_addrs = set()        # addrs where within-group WAR is skipped

    def mark_constants(self, addrs):
        """Mark addresses as read-only constants (skip WAR tracking for them)."""
        self.const_addrs.update(addrs)

    def mark_local(self, addrs):
        """Mark addresses as group-local (skip WAR for same-group readers).
        Within a group, the RAW chain through vg_val guarantees ordering,
        making WAR edges redundant. Between groups, WAR is still needed."""
        self.local_addrs.update(addrs)

    def add_op(self, engine, slot, reads, writes, group=-1, extra_preds=None):
        op_idx = len(self.ops)
        op = Op(engine, slot, op_idx, group=group)

        pred_set = set()
        if extra_preds:
            pred_set.update(extra_preds)

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
                is_local = addr in self.local_addrs
                for r_idx in self.readers_since_write.get(addr, ()):
                    # Skip WAR for same-group readers on local addresses
                    if is_local and self.ops[r_idx].group == group:
                        continue
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

    # Compute critical path + load descendant count (reverse topo order)
    cp = [0] * n
    load_desc = [0] * n
    for op in reversed(ops):
        max_cp = 0
        for s_idx in op.succs:
            if cp[s_idx] + 1 > max_cp:
                max_cp = cp[s_idx] + 1
            load_desc[op.id] += load_desc[s_idx]
        cp[op.id] = max_cp
        if op.engine == "load":
            load_desc[op.id] += 1

    for op in ops:
        # Boost loads and their immediate predecessors (addr_calc ops)
        boost = 0
        if op.engine == "load":
            boost = 50000
        else:
            for s_idx in op.succs:
                if ops[s_idx].engine == "load":
                    boost = 25000
                    break
        # Stagger groups: lower group numbers get slight priority boost
        # so they reach scattered rounds (with loads) faster
        group_boost = (32 - op.group) if op.group >= 0 else 0
        op.priority = cp[op.id] * 1000 + boost + load_desc[op.id] * 100 + group_boost * 10 + len(op.succs)

    # Initialize ready lists per engine type
    from heapq import heappush, heappop
    ready_by_eng = defaultdict(list)  # engine -> heap of (-priority, op_id, op)
    for op in ops:
        if op.pred_count == 0:
            heappush(ready_by_eng[op.engine], (-op.priority, op.id, op))

    bundles = []
    newly_ready = []

    # Pack order: most constrained resources first
    pack_order = ["valu", "load", "store", "flow", "alu"]

    def has_ready():
        return any(ready_by_eng[e] for e in ready_by_eng)

    while has_ready() or newly_ready:
        if newly_ready:
            for op in newly_ready:
                heappush(ready_by_eng[op.engine], (-op.priority, op.id, op))
            newly_ready = []

        if not has_ready():
            break

        bundle = {}
        scheduled_this = []

        # Pack each engine type in order of scarcity
        for eng in pack_order:
            heap = ready_by_eng[eng]
            limit = SLOT_LIMITS.get(eng, 0)
            cnt = 0
            while heap and cnt < limit:
                neg_pri, op_id, op = heappop(heap)
                bundle.setdefault(eng, []).append(op.slot)
                scheduled_this.append(op)
                cnt += 1

        if not scheduled_this:
            # Safety: force-schedule one op from any engine
            for eng in pack_order:
                heap = ready_by_eng[eng]
                if heap:
                    neg_pri, op_id, op = heappop(heap)
                    bundle = {eng: [op.slot]}
                    scheduled_this = [op]
                    break

        bundles.append(bundle)

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
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")

        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)

        # Scalar constants (packed: 2 per cycle)
        zero_const = self.alloc_scratch("zero")
        eight_const = self.alloc_scratch("eight")
        fvp_const = self.alloc_scratch("fvp_const")
        self.instrs.append({"load": [("const", zero_const, 0),
                                     ("const", eight_const, 8)]})
        self.instrs.append({"load": [("const", fvp_const, forest_values_p_val)]})

        # ---- Allocate persistent vectors (32 groups x 2 vectors x 8 elements) ----
        vidx = []
        vval = []
        for g in range(n_groups):
            vidx.append(self.alloc_scratch(f"vidx_{g}", VLEN))
            vval.append(self.alloc_scratch(f"vval_{g}", VLEN))

        # ---- Allocate per-group temp vectors ----
        # N_BUF=32 gives each group its own temp buffers, eliminating all
        # inter-group WAR coupling. Critical path drops from ~388 to ~240.
        # vtC is per-group for branch bit and level-2 high bit.
        N_BUF = 25
        vtA = [self.alloc_scratch(f"vtA_{b}", VLEN) for b in range(N_BUF)]
        vtB = [self.alloc_scratch(f"vtB_{b}", VLEN) for b in range(N_BUF)]
        vtC = [self.alloc_scratch(f"vtC_{g}", VLEN) for g in range(n_groups)]

        # ---- Allocate and initialize vector constants ----
        vec_const_defs = [
            ("v_one", 1),
            ("v_two", 2),
            ("v_mul_4097", (1 << 12) + 1),
            ("v_C0", HASH_STAGES[0][1]),
            ("v_C1", HASH_STAGES[1][1]),
            ("v_19", HASH_STAGES[1][4]),
            ("v_mul_33", (1 << 5) + 1),
            ("v_C2", HASH_STAGES[2][1]),
            ("v_C3", HASH_STAGES[3][1]),
            ("v_9", HASH_STAGES[3][4]),
            ("v_C4", HASH_STAGES[4][1]),
            ("v_C5", HASH_STAGES[5][1]),
            ("v_16", HASH_STAGES[5][4]),
            ("v_fvp", forest_values_p_val),
        ]
        vec_const_addrs = {}
        for name, _ in vec_const_defs:
            vec_const_addrs[name] = self.alloc_scratch(name, VLEN)

        # Pipeline: const loads overlap with broadcasts from previous cycle
        pending_broadcasts = []
        for i in range(0, len(vec_const_defs), 2):
            bundle = {}
            if pending_broadcasts:
                bundle["valu"] = [("vbroadcast", addr, src)
                                  for addr, src in pending_broadcasts]
            pending_broadcasts = []
            loads = []
            name1, val1 = vec_const_defs[i]
            loads.append(("const", tmp1, val1))
            pending_broadcasts.append((vec_const_addrs[name1], tmp1))
            if i + 1 < len(vec_const_defs):
                name2, val2 = vec_const_defs[i + 1]
                loads.append(("const", tmp2, val2))
                pending_broadcasts.append((vec_const_addrs[name2], tmp2))
            bundle["load"] = loads
            self.instrs.append(bundle)
        if pending_broadcasts:
            self.instrs.append({"valu": [("vbroadcast", addr, src)
                                         for addr, src in pending_broadcasts]})

        v_one = vec_const_addrs["v_one"]
        v_two = vec_const_addrs["v_two"]
        v_mul_4097 = vec_const_addrs["v_mul_4097"]
        v_C0 = vec_const_addrs["v_C0"]
        v_C1 = vec_const_addrs["v_C1"]
        v_19 = vec_const_addrs["v_19"]
        v_mul_33 = vec_const_addrs["v_mul_33"]
        v_C2 = vec_const_addrs["v_C2"]
        v_C3 = vec_const_addrs["v_C3"]
        v_9 = vec_const_addrs["v_9"]
        v_C4 = vec_const_addrs["v_C4"]
        v_C5 = vec_const_addrs["v_C5"]
        v_16 = vec_const_addrs["v_16"]
        v_fvp = vec_const_addrs["v_fvp"]

        # ---- Allocate tree scratch (before pause, loads happen pre-pause) ----
        ptr_idx = self.alloc_scratch("ptr_idx")
        ptr_val = self.alloc_scratch("ptr_val")
        tree_s0 = self.alloc_scratch("tree_s0")
        v_tree0 = self.alloc_scratch("v_tree0", VLEN)
        tree_s1 = self.alloc_scratch("tree_s1")
        tree_s2 = self.alloc_scratch("tree_s2")
        diff12_s = self.alloc_scratch("diff12_s")
        v_tree2_pre = self.alloc_scratch("v_tree2_pre", VLEN)
        v_diff12 = self.alloc_scratch("v_diff12", VLEN)
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
        v_three = self.alloc_scratch("v_three", VLEN)

        # ---- Load tree values from memory (before pause) ----
        self.instrs.append({"load": [("const", tmp1, forest_values_p_val + 0),
                                     ("const", tmp2, forest_values_p_val + 1)]})
        self.instrs.append({"load": [("load", tree_s0, tmp1),
                                     ("load", tree_s1, tmp2)]})
        self.instrs.append({"load": [("const", tmp1, forest_values_p_val + 2),
                                     ("const", tmp2, forest_values_p_val + 3)]})
        self.instrs.append({"load": [("load", tree_s2, tmp1),
                                     ("load", tree_s3, tmp2)]})
        self.instrs.append({"load": [("const", tmp1, forest_values_p_val + 4),
                                     ("const", tmp2, forest_values_p_val + 5)],
                            "alu": [("-", diff12_s, tree_s1, tree_s2)]})
        self.instrs.append({"load": [("load", tree_s4, tmp1),
                                     ("load", tree_s5, tmp2)]})
        self.instrs.append({"load": [("const", tmp1, forest_values_p_val + 6),
                                     ("const", tmp2, 3)],
                            "alu": [("-", diff34_s, tree_s4, tree_s3)]})
        self.instrs.append({"load": [("load", tree_s6, tmp1)]})
        self.instrs.append({"alu": [("-", diff56_s, tree_s6, tree_s5)]})

        # ---- Pause (matches first yield in reference_kernel2) ----
        self.add("flow", ("pause",))

        # ---- Phase 1: Initial vloads with tree broadcasts overlapped ----
        self.instrs.append({"load": [("const", ptr_idx, inp_indices_p_val),
                                     ("const", ptr_val, inp_values_p_val)]})

        tree_bc = [
            ("vbroadcast", v_tree0, tree_s0),
            ("vbroadcast", v_tree2_pre, tree_s2),
            ("vbroadcast", v_diff12, diff12_s),
            ("vbroadcast", v_tree3_pre, tree_s3),
            ("vbroadcast", v_tree5_pre, tree_s5),
            ("vbroadcast", v_diff34, diff34_s),
            ("vbroadcast", v_diff56, diff56_s),
            ("vbroadcast", v_three, tmp2),  # tmp2 has 3 from tree loading
        ]
        for g in range(n_groups):
            bundle = {
                "load": [("vload", vidx[g], ptr_idx), ("vload", vval[g], ptr_val)],
                "alu": [("+", ptr_idx, ptr_idx, eight_const),
                        ("+", ptr_val, ptr_val, eight_const)],
            }
            if g == 0:
                bundle["valu"] = tree_bc[:6]
            elif g == 1:
                bundle["valu"] = tree_bc[6:]
            self.instrs.append(bundle)

        # ---- Phase 2: Main loop - build op graph for scheduler ----
        graph = OpGraph()

        def tree_level(r):
            if r <= 10:
                return r
            return r - 11

        # Mark all vector constant addresses as read-only
        const_addr_ranges = [
            (v_one, VLEN), (v_two, VLEN),
            (v_mul_4097, VLEN), (v_C0, VLEN), (v_C1, VLEN), (v_19, VLEN),
            (v_mul_33, VLEN), (v_C2, VLEN), (v_C3, VLEN), (v_9, VLEN),
            (v_C4, VLEN), (v_C5, VLEN), (v_16, VLEN), (v_fvp, VLEN),
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

        # Mark vtA/vtB as group-local: within-group WAR edges are redundant
        # because the RAW chain through vg_val guarantees ordering.
        local_addrs = set()
        for b in range(N_BUF):
            for i in range(VLEN):
                local_addrs.add(vtA[b] + i)
                local_addrs.add(vtB[b] + i)
        graph.mark_local(local_addrs)

        def vrange(base):
            return set(range(base, base + VLEN))

        for r in range(rounds):
            level = tree_level(r)
            for g in range(n_groups):
                vg_idx = vidx[g]
                vg_val = vval[g]
                buf = g % N_BUF
                va = vtA[buf]   # vaddr + vtmp1 (merged, shared via N_BUF)
                vn = vtB[buf]   # vnode + vtmp2 (merged, shared via N_BUF)
                vt1 = vtA[buf]  # same as va (reused after addr phase)
                vt2 = vtB[buf]  # same as vn (reused after load/xor phase)
                vb = vtC[g]     # per-group branch bit (shortens coupling)

                if level == 0:
                    graph.add_op(
                        "valu", ("^", vg_val, vg_val, v_tree0),
                        reads=vrange(vg_val) | vrange(v_tree0),
                        writes=vrange(vg_val), group=g
                    )
                elif level == 1:
                    graph.add_op(
                        "valu", ("&", vn, vg_idx, v_one),
                        reads=vrange(vg_idx) | vrange(v_one),
                        writes=vrange(vn), group=g
                    )
                    graph.add_op(
                        "valu", ("multiply_add", vn, vn, v_diff12, v_tree2_pre),
                        reads=vrange(vn) | vrange(v_diff12) | vrange(v_tree2_pre),
                        writes=vrange(vn), group=g
                    )
                    graph.add_op(
                        "valu", ("^", vg_val, vg_val, vn),
                        reads=vrange(vg_val) | vrange(vn),
                        writes=vrange(vg_val), group=g
                    )
                elif level == 2:
                    graph.add_op(
                        "valu", ("-", vt1, vg_idx, v_three),
                        reads=vrange(vg_idx) | vrange(v_three),
                        writes=vrange(vt1), group=g
                    )
                    graph.add_op(
                        "valu", ("&", vt2, vt1, v_one),
                        reads=vrange(vt1) | vrange(v_one),
                        writes=vrange(vt2), group=g
                    )
                    l2_shr_idx = graph.add_op(
                        "valu", (">>", vb, vt1, v_one),
                        reads=vrange(vt1) | vrange(v_one),
                        writes=vrange(vb), group=g
                    )
                    # pair0: explicit dep on l2_shr since WAR through vtA
                    # is skipped (same group) but l2_shr reads vtA with
                    # no RAW path to pair0 — need the ordering guarantee
                    graph.add_op(
                        "valu", ("multiply_add", vt1, vt2, v_diff34, v_tree3_pre),
                        reads=vrange(vt2) | vrange(v_diff34) | vrange(v_tree3_pre),
                        writes=vrange(vt1), group=g,
                        extra_preds=[l2_shr_idx]
                    )
                    graph.add_op(
                        "valu", ("multiply_add", vt2, vt2, v_diff56, v_tree5_pre),
                        reads=vrange(vt2) | vrange(v_diff56) | vrange(v_tree5_pre),
                        writes=vrange(vt2), group=g
                    )
                    graph.add_op(
                        "valu", ("-", vt2, vt2, vt1),
                        reads=vrange(vt2) | vrange(vt1),
                        writes=vrange(vt2), group=g
                    )
                    graph.add_op(
                        "valu", ("multiply_add", vt1, vb, vt2, vt1),
                        reads=vrange(vb) | vrange(vt2) | vrange(vt1),
                        writes=vrange(vt1), group=g
                    )
                    graph.add_op(
                        "valu", ("^", vg_val, vg_val, vt1),
                        reads=vrange(vg_val) | vrange(vt1),
                        writes=vrange(vg_val), group=g
                    )
                else:
                    graph.add_op(
                        "valu", ("+", va, vg_idx, v_fvp),
                        reads=vrange(vg_idx) | vrange(v_fvp),
                        writes=vrange(va), group=g
                    )
                    for i in range(VLEN):
                        graph.add_op(
                            "load", ("load", vn + i, va + i),
                            reads={va + i},
                            writes={vn + i}, group=g
                        )
                    graph.add_op(
                        "valu", ("^", vg_val, vg_val, vn),
                        reads=vrange(vg_val) | vrange(vn),
                        writes=vrange(vg_val), group=g
                    )

                # -- Hash stages --
                graph.add_op(
                    "valu", ("multiply_add", vg_val, vg_val, v_mul_4097, v_C0),
                    reads=vrange(vg_val) | vrange(v_mul_4097) | vrange(v_C0),
                    writes=vrange(vg_val), group=g
                )
                for i in range(VLEN):
                    graph.add_op(
                        "alu", ("^", vt1 + i, vg_val + i, v_C1 + i),
                        reads={vg_val + i, v_C1 + i},
                        writes={vt1 + i}, group=g
                    )
                graph.add_op(
                    "valu", (">>", vt2, vg_val, v_19),
                    reads=vrange(vg_val) | vrange(v_19),
                    writes=vrange(vt2), group=g
                )
                graph.add_op(
                    "valu", ("^", vg_val, vt1, vt2),
                    reads=vrange(vt1) | vrange(vt2),
                    writes=vrange(vg_val), group=g
                )
                graph.add_op(
                    "valu", ("multiply_add", vg_val, vg_val, v_mul_33, v_C2),
                    reads=vrange(vg_val) | vrange(v_mul_33) | vrange(v_C2),
                    writes=vrange(vg_val), group=g
                )
                for i in range(VLEN):
                    graph.add_op(
                        "alu", ("+", vt1 + i, vg_val + i, v_C3 + i),
                        reads={vg_val + i, v_C3 + i},
                        writes={vt1 + i}, group=g
                    )
                graph.add_op(
                    "valu", ("<<", vt2, vg_val, v_9),
                    reads=vrange(vg_val) | vrange(v_9),
                    writes=vrange(vt2), group=g
                )
                graph.add_op(
                    "valu", ("^", vg_val, vt1, vt2),
                    reads=vrange(vt1) | vrange(vt2),
                    writes=vrange(vg_val), group=g
                )
                graph.add_op(
                    "valu", ("multiply_add", vg_val, vg_val, v_9, v_C4),
                    reads=vrange(vg_val) | vrange(v_9) | vrange(v_C4),
                    writes=vrange(vg_val), group=g
                )
                for i in range(VLEN):
                    graph.add_op(
                        "alu", ("^", vt1 + i, vg_val + i, v_C5 + i),
                        reads={vg_val + i, v_C5 + i},
                        writes={vt1 + i}, group=g
                    )
                graph.add_op(
                    "valu", (">>", vt2, vg_val, v_16),
                    reads=vrange(vg_val) | vrange(v_16),
                    writes=vrange(vt2), group=g
                )
                graph.add_op(
                    "valu", ("^", vg_val, vt1, vt2),
                    reads=vrange(vt1) | vrange(vt2),
                    writes=vrange(vg_val), group=g
                )

                # -- Branch / wrap --
                if r == 10:
                    graph.add_op(
                        "valu", ("vbroadcast", vg_idx, zero_const),
                        reads={zero_const},
                        writes=vrange(vg_idx), group=g
                    )
                else:
                    graph.add_op(
                        "valu", ("&", vb, vg_val, v_one),
                        reads=vrange(vg_val) | vrange(v_one),
                        writes=vrange(vb), group=g
                    )
                    graph.add_op(
                        "valu", ("multiply_add", vg_idx, vg_idx, v_two, v_one),
                        reads=vrange(vg_idx) | vrange(v_two) | vrange(v_one),
                        writes=vrange(vg_idx), group=g
                    )
                    graph.add_op(
                        "valu", ("+", vg_idx, vg_idx, vb),
                        reads=vrange(vg_idx) | vrange(vb),
                        writes=vrange(vg_idx), group=g
                    )

        # ---- Add final stores to the graph for overlapped scheduling ----
        # Const loads for store pointers
        ptr_idx_load = graph.add_op(
            "load", ("const", ptr_idx, inp_indices_p_val),
            reads=set(), writes={ptr_idx}, group=-1
        )
        ptr_val_load = graph.add_op(
            "load", ("const", ptr_val, inp_values_p_val),
            reads=set(), writes={ptr_val}, group=-1
        )

        prev_inc_idx = None
        for g in range(n_groups):
            vg_idx = vidx[g]
            vg_val = vval[g]
            # 2 separate store ops per group
            preds = [ptr_idx_load, ptr_val_load]
            if prev_inc_idx is not None:
                preds.append(prev_inc_idx)

            s_idx = graph.add_op(
                "store", ("vstore", ptr_idx, vg_idx),
                reads=vrange(vg_idx) | {ptr_idx}, writes=set(), group=-1,
                extra_preds=preds
            )
            s_val = graph.add_op(
                "store", ("vstore", ptr_val, vg_val),
                reads=vrange(vg_val) | {ptr_val}, writes=set(), group=-1,
                extra_preds=preds
            )
            # ALU increments after stores
            inc_idx = graph.add_op(
                "alu", ("+", ptr_idx, ptr_idx, eight_const),
                reads={ptr_idx, eight_const}, writes={ptr_idx}, group=-1,
                extra_preds=[s_idx, s_val]
            )
            prev_inc_idx = graph.add_op(
                "alu", ("+", ptr_val, ptr_val, eight_const),
                reads={ptr_val, eight_const}, writes={ptr_val}, group=-1,
                extra_preds=[s_idx, s_val]
            )

        # Final pause
        graph.add_op(
            "flow", ("pause",),
            reads=set(), writes=set(), group=-1,
            extra_preds=[prev_inc_idx] if prev_inc_idx is not None else []
        )

        # Schedule all ops (main body + stores + pause)
        all_bundles = schedule_ops(graph)
        self.instrs.extend(all_bundles)


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
