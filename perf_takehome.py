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


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

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

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Optimized kernel.

        Notes:
        - This version is written to be fast on the simulated VLIW/SIMD machine.
        - It keeps `idx` and `val` in scratch vectors for the whole kernel and
          writes back only once at the end.
        - It uses `valu.multiply_add` to compress 3 of the 6 hash stages.
        - For general depths it still gathers node values (no vgather ISA), but
          schedules loads and valu work together to hit the load/valu rooflines.

        This is not necessarily the global optimum; it's a solid, readable
        baseline for further iteration.
        """
        assert batch_size % VLEN == 0, "This kernel assumes batch_size is a multiple of VLEN"

        # --- Scalar scratch regs (addresses, constants) ---
        s_forest_base = self.alloc_scratch("forest_values_p")
        s_idx_base = self.alloc_scratch("inp_indices_p")
        s_val_base = self.alloc_scratch("inp_values_p")

        # These are fixed by build_mem_image's layout: header is 7 words.
        header_words = 7
        forest_values_p = header_words
        inp_indices_p = forest_values_p + n_nodes
        inp_values_p = inp_indices_p + batch_size

        # Common scalar constants
        s_zero = self.alloc_scratch("c0")
        s_one = self.alloc_scratch("c1")
        s_two = self.alloc_scratch("c2")
        s_eight = self.alloc_scratch("c8")

        # Hash constants (scalars)
        s_mul4097 = self.alloc_scratch("mul4097")
        s_mul33 = self.alloc_scratch("mul33")
        s_mul9 = self.alloc_scratch("mul9")
        s_c1 = self.alloc_scratch("h_c1")
        s_c2 = self.alloc_scratch("h_c2")
        s_c3 = self.alloc_scratch("h_c3")
        s_c4 = self.alloc_scratch("h_c4")
        s_c5 = self.alloc_scratch("h_c5")
        s_c6 = self.alloc_scratch("h_c6")
        s_sh19 = self.alloc_scratch("sh19")
        s_sh9 = self.alloc_scratch("sh9")
        s_sh16 = self.alloc_scratch("sh16")

        # --- Vector scratch regs (constants) ---
        def alloc_vec(name: str):
            return self.alloc_scratch(name, VLEN)

        v_forest_base = alloc_vec("v_forest_base")
        v_zero = alloc_vec("v0")
        v_one = alloc_vec("v1")
        v_two = alloc_vec("v2")
        v_mul4097 = alloc_vec("v_mul4097")
        v_mul33 = alloc_vec("v_mul33")
        v_mul9 = alloc_vec("v_mul9")
        v_c1 = alloc_vec("v_h_c1")
        v_c2 = alloc_vec("v_h_c2")
        v_c3 = alloc_vec("v_h_c3")
        v_c4 = alloc_vec("v_h_c4")
        v_c5 = alloc_vec("v_h_c5")
        v_c6 = alloc_vec("v_h_c6")
        v_sh19 = alloc_vec("v_sh19")
        v_sh9 = alloc_vec("v_sh9")
        v_sh16 = alloc_vec("v_sh16")

        # --- Per-group vectors (idx, val, node, temps) ---
        n_groups = batch_size // VLEN
        v_idx = self.alloc_scratch("idx", batch_size)
        v_val = self.alloc_scratch("val", batch_size)
        v_node = self.alloc_scratch("node", batch_size)
        v_tmp = self.alloc_scratch("tmp", batch_size)
        v_tmp2 = self.alloc_scratch("tmp2", batch_size)

        def idx_base_of(g: int) -> int:
            return v_idx + g * VLEN

        def val_base_of(g: int) -> int:
            return v_val + g * VLEN

        def node_base_of(g: int) -> int:
            return v_node + g * VLEN

        def tmp_base_of(g: int) -> int:
            return v_tmp + g * VLEN

        def tmp2_base_of(g: int) -> int:
            return v_tmp2 + g * VLEN

        # --- VLIW emit helpers ---
        def emit_cycle(
            alu_slots=None, valu_slots=None, load_slots=None, store_slots=None, flow_slots=None
        ):
            instr = {}
            if alu_slots:
                instr["alu"] = alu_slots
            if valu_slots:
                instr["valu"] = valu_slots
            if load_slots:
                instr["load"] = load_slots
            if store_slots:
                instr["store"] = store_slots
            if flow_slots:
                instr["flow"] = flow_slots
            if not instr:
                return
            self.instrs.append(instr)

        # --- Initialization ---
        # Required to match the initial yield in reference_kernel2.
        emit_cycle(flow_slots=[("pause",)])

        # Load scalar constants and pointers.
        # Each cycle can issue 2 load slots.
        const_pairs = [
            (s_forest_base, forest_values_p),
            (s_idx_base, inp_indices_p),
            (s_val_base, inp_values_p),
            (s_zero, 0),
            (s_one, 1),
            (s_two, 2),
            (s_eight, VLEN),
            (s_mul4097, 1 + (1 << 12)),
            (s_mul33, 1 + (1 << 5)),
            (s_mul9, 1 + (1 << 3)),
            (s_c1, 0x7ED55D16),
            (s_c2, 0xC761C23C),
            (s_c3, 0x165667B1),
            (s_c4, 0xD3A2646C),
            (s_c5, 0xFD7046C5),
            (s_c6, 0xB55A4F09),
            (s_sh19, 19),
            (s_sh9, 9),
            (s_sh16, 16),
        ]
        i = 0
        while i < len(const_pairs):
            loads = [("const", const_pairs[i][0], const_pairs[i][1])]
            i += 1
            if i < len(const_pairs):
                loads.append(("const", const_pairs[i][0], const_pairs[i][1]))
                i += 1
            emit_cycle(load_slots=loads)

        # Broadcast scalar constants to vectors (6 valu slots/cycle).
        broadcasts = [
            (v_forest_base, s_forest_base),
            (v_zero, s_zero),
            (v_one, s_one),
            (v_two, s_two),
            (v_mul4097, s_mul4097),
            (v_mul33, s_mul33),
            (v_mul9, s_mul9),
            (v_c1, s_c1),
            (v_c2, s_c2),
            (v_c3, s_c3),
            (v_c4, s_c4),
            (v_c5, s_c5),
            (v_c6, s_c6),
            (v_sh19, s_sh19),
            (v_sh9, s_sh9),
            (v_sh16, s_sh16),
        ]
        j = 0
        while j < len(broadcasts):
            vs = []
            for _ in range(6):
                if j >= len(broadcasts):
                    break
                dest, src = broadcasts[j]
                vs.append(("vbroadcast", dest, src))
                j += 1
            emit_cycle(valu_slots=vs)

        # Load initial idx/val vectors from memory.
        # We'll stream with two scalar pointers in scratch and vload both each cycle.
        s_pidx = self.alloc_scratch("pidx")
        s_pval = self.alloc_scratch("pval")
        emit_cycle(load_slots=[("const", s_pidx, inp_indices_p), ("const", s_pval, inp_values_p)])

        for g in range(n_groups):
            emit_cycle(
                load_slots=[
                    ("vload", idx_base_of(g), s_pidx),
                    ("vload", val_base_of(g), s_pval),
                ],
                alu_slots=[
                    ("+", s_pidx, s_pidx, s_eight),
                    ("+", s_pval, s_pval, s_eight),
                ],
            )

        # --- Per-step scheduling ---
        def group_stage_ops(g: int, stage: int, node_src_vec: int, leaf_wrap: bool):
            """
            Return a list of 1-2 valu slots for this group stage.
            Stages are structured so that slots within a stage are independent
            (read same prior values, write distinct dests).
            """
            vb = val_base_of(g)
            ib = idx_base_of(g)
            nb = node_src_vec
            tb = tmp_base_of(g)
            t2b = tmp2_base_of(g)

            # Hash + idx update, using tmp/tmp2 as scratch.
            if stage == 0:
                return [("^", vb, vb, nb)]
            if stage == 1:
                return [("multiply_add", vb, vb, v_mul4097, v_c1)]
            if stage == 2:
                return [(">>", tb, vb, v_sh19), ("^", t2b, vb, v_c2)]
            if stage == 3:
                return [("^", vb, t2b, tb)]
            if stage == 4:
                return [("multiply_add", vb, vb, v_mul33, v_c3)]
            if stage == 5:
                return [("<<", tb, vb, v_sh9), ("+", t2b, vb, v_c4)]
            if stage == 6:
                return [("^", vb, t2b, tb)]
            if stage == 7:
                return [("multiply_add", vb, vb, v_mul9, v_c5)]
            if stage == 8:
                return [(">>", tb, vb, v_sh16), ("^", t2b, vb, v_c6)]
            if stage == 9:
                return [("^", vb, t2b, tb)]

            if leaf_wrap:
                # After processing a leaf, we always wrap idx to 0 in a perfect
                # binary tree traversal (children are out of range).
                if stage == 10:
                    return [("vbroadcast", ib, s_zero)]
                raise AssertionError("bad stage for leaf_wrap")

            if stage == 10:
                return [("&", t2b, vb, v_one)]
            if stage == 11:
                return [("+", t2b, t2b, v_one)]
            if stage == 12:
                return [("multiply_add", ib, ib, v_two, t2b)]
            raise AssertionError("bad stage")

        def schedule_step(depth: int, leaf_wrap: bool):
            # Depth 0: idx is 0 for all lanes, so node value is constant.
            use_const_node0 = depth == 0
            node0_vec = v_node0  # set below (after we load it once)

            # Per-group state.
            # load_phase:
            #   0 = need addr-calc (node = idx + forest_base) in valu
            #   1..4 = load_offset pairs (2 lanes/cycle) from node's per-lane addresses
            #   5 = node values loaded and ready for compute
            load_phase = [5 if use_const_node0 else 0] * n_groups
            stage = [0] * n_groups

            # Next group to advance through gather.
            g_load = 0  # points at group currently being loaded (phase 0..4), then advances
            g_addr = 0  # points at next group needing addr-calc
            rr = 0  # round-robin start for valu
            n_stages = 11 if leaf_wrap else 13

            while True:
                if all(s >= n_stages for s in stage):
                    break

                alu_slots = []
                load_slots = []
                valu_slots = []
                no_compute = set()

                # Gather 2 lanes per cycle for one group (load engine is limiting).
                if not use_const_node0:
                    addr_calc_g = None
                    # Issue one addr-calc each cycle if available, using 1 valu slot.
                    # This must happen at least 1 cycle before the first load_offset uses it.
                    while g_addr < n_groups and load_phase[g_addr] != 0:
                        g_addr += 1
                    if g_addr < n_groups:
                        # node_vec := idx_vec + forest_base (per-lane address into mem)
                        addr_calc_g = g_addr
                        valu_slots.append(("+", node_base_of(g_addr), idx_base_of(g_addr), v_forest_base))
                        load_phase[g_addr] = 1
                        g_addr += 1

                    # Advance load_phase for one group by loading 2 offsets.
                    while g_load < n_groups and load_phase[g_load] == 5:
                        g_load += 1
                    # Do not issue load_offset for the same group whose addresses we
                    # computed this cycle; loads would see the previous addresses.
                    if g_load == addr_calc_g:
                        pass
                    elif g_load < n_groups and 1 <= load_phase[g_load] <= 4:
                        no_compute.add(g_load)
                        pair = load_phase[g_load] - 1  # 0..3
                        off0 = pair * 2
                        off1 = off0 + 1
                        nb = node_base_of(g_load)
                        load_slots.append(("load_offset", nb, nb, off0))
                        load_slots.append(("load_offset", nb, nb, off1))
                        load_phase[g_load] += 1
                        if load_phase[g_load] == 5:
                            g_load += 1

                # Schedule valu work: greedily fill up to 6 valu slots.
                slots_left = 6 - len(valu_slots)
                tried = 0
                while slots_left > 0 and tried < n_groups:
                    g = rr % n_groups
                    rr += 1
                    tried += 1
                    if load_phase[g] != 5:
                        continue
                    if g in no_compute:
                        continue
                    if stage[g] >= n_stages:
                        continue

                    node_src = node0_vec if use_const_node0 else node_base_of(g)
                    ops = group_stage_ops(g, stage[g], node_src, leaf_wrap)
                    if len(ops) > slots_left:
                        continue
                    valu_slots.extend(ops)
                    stage[g] += 1
                    slots_left -= len(ops)

                emit_cycle(
                    alu_slots=alu_slots,
                    load_slots=load_slots,
                    valu_slots=valu_slots,
                )

        # Load forest[0] once and build a reusable node0 vector.
        s_node0 = self.alloc_scratch("node0")
        v_node0 = alloc_vec("v_node0")
        emit_cycle(load_slots=[("load", s_node0, s_forest_base)])
        emit_cycle(valu_slots=[("vbroadcast", v_node0, s_node0)])

        for step in range(rounds):
            depth = step % (forest_height + 1) if forest_height >= 0 else 0
            leaf_wrap = depth == forest_height and forest_height >= 0
            schedule_step(depth=depth, leaf_wrap=leaf_wrap)

        # Store idx/val back to memory once.
        emit_cycle(load_slots=[("const", s_pidx, inp_indices_p), ("const", s_pval, inp_values_p)])
        for g in range(n_groups):
            emit_cycle(
                store_slots=[
                    ("vstore", s_pidx, idx_base_of(g)),
                    ("vstore", s_pval, val_base_of(g)),
                ],
                alu_slots=[
                    ("+", s_pidx, s_pidx, s_eight),
                    ("+", s_pval, s_pval, s_eight),
                ],
            )

        # Required to match with the final yield in reference_kernel2.
        emit_cycle(flow_slots=[("pause",)])

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
    # print(kb.instrs)

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
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
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
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
