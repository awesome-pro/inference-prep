"""
engine_torch.py

The engine from engine.py, but with a REAL worker: actual tensors, an actual
paged KV cache, and actual paged attention.

Read engine.py first. This file keeps its scheduler unchanged on purpose.

=====================================================================
THE POINT: lists on the CPU side, tensors on the GPU side
=====================================================================
This is the thing people get wrong. In vLLM the split is not "we switched
everything to torch". It is:

    scheduler / manager   ->  Python lists     (control plane)
    worker / model runner ->  torch tensors    (data plane)

Verified in the real source:
    Request._all_token_ids        list[int]        (v1/request.py)
    req_to_blocks                 list[KVCacheBlock] (single_type_kv_cache_manager.py)
    self.input_ids                torch buffer     (gpu_model_runner.py:765)
    token_ids_cpu                 torch -> .numpy() (gpu_input_batch.py:131)

Why keep the CPU side as lists?
  * The scheduler's job is branching and bookkeeping, not arithmetic. A Python
    int per token is faster to append and to hash than a tensor slice.
  * Preemption must clear a block list and zero a counter. Trivial with lists;
    awkward with tensors you would have to invalidate.
  * Token tensors are built ONCE per step, for the whole batch, in one place.
    That is the only place they need to exist.

=====================================================================
WHAT THE WORKER DOES THAT engine.py DID NOT
=====================================================================
1. PRE-ALLOCATED BUFFERS. `input_ids` is allocated once at max size and
   written into a slice each step. Never allocate in the loop.
2. SLOT MAPPING.  slot = block_id * block_size + offset, one int per token.
   This is the bridge between the allocator and the attention kernel.
3. THE KV CACHE IS ONE TENSOR. Writing a step is a single scatter:
       cache.view(-1, 2, D)[slot_mapping] = k_new, v_new
4. PAGED ATTENTION. The kernel walks the block table. It never builds a
   contiguous [seq, D] tensor -- that is the whole point of paging.
5. ONLY THE LAST POSITION IS SAMPLED. The model computes all positions, but
   we only need logits at the end of the sequence.
"""

from collections import deque
from dataclasses import dataclass, field

import torch

HEAD_DIM = 8
VOCAB = 8


# ======================================================================
# 1. memory -- unchanged from engine.py, still just lists of ints
# ======================================================================
class Block:
    __slots__ = ("block_id", "ref_count", "hash")

    def __init__(self, block_id: int) -> None:
        self.block_id = block_id
        self.ref_count = 0
        self.hash: int | None = None


class KVCacheManager:
    """Scheduler-side memory bookkeeping. Holds NO tensors."""

    def __init__(self, num_blocks: int, block_size: int) -> None:
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.free_blocks: deque[int] = deque(range(num_blocks))
        self.cached: dict[int, Block] = {}
        self.req_to_blocks: dict[str, list[int]] = {}

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def _take(self) -> int:
        block = self.blocks[self.free_blocks.popleft()]
        if block.hash is not None:
            del self.cached[block.hash]
            block.hash = None
        block.ref_count = 1
        return block.block_id

    def _release(self, block_id: int) -> None:
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_blocks.append(block_id)

    def get_computed_blocks(self, request) -> tuple[list[int], int]:
        blocks: list[int] = []
        parent = None
        n = 0
        while n + self.block_size <= len(request.tokens) - 1:
            key = hash((parent, tuple(request.tokens[n : n + self.block_size])))
            block = self.cached.get(key)
            if block is None:
                break
            if block.ref_count == 0:
                self.free_blocks.remove(block.block_id)
            block.ref_count += 1
            blocks.append(block.block_id)
            parent = key
            n += self.block_size
        return blocks, n

    def allocate_slots(self, request, num_new_tokens: int,
                       adopted: list[int] | None = None) -> list[int] | None:
        adopted = adopted or []
        num_computed = request.num_computed_tokens + len(adopted) * self.block_size
        required = -(-(num_computed + num_new_tokens) // self.block_size)
        existing = len(self.req_to_blocks.get(request.req_id, []))
        num_new = required - existing - len(adopted)

        if num_new > self.num_free_blocks:
            for block_id in adopted:
                self._release(block_id)
            return None

        allocated = list(adopted)
        for _ in range(num_new):
            allocated.append(self._take())
        self.req_to_blocks.setdefault(request.req_id, []).extend(allocated)
        self._cache_full_blocks(request, num_computed)
        return allocated

    def _cache_full_blocks(self, request, up_to: int) -> None:
        parent = None
        for i, block_id in enumerate(self.req_to_blocks[request.req_id]):
            end = (i + 1) * self.block_size
            if end > up_to:
                break
            key = hash((parent, tuple(request.tokens[end - self.block_size : end])))
            block = self.blocks[block_id]
            if block.hash is None and key not in self.cached:
                block.hash = key
                self.cached[key] = block
            parent = key

    def free(self, request) -> None:
        for block_id in self.req_to_blocks.pop(request.req_id, []):
            self._release(block_id)

    def block_table(self, request) -> list[int]:
        return list(self.req_to_blocks.get(request.req_id, []))


# ======================================================================
# 2. request + plan -- unchanged
# ======================================================================
WAITING, RUNNING, PREEMPTED, DONE = "waiting", "running", "preempted", "done"


@dataclass
class Request:
    req_id: str
    tokens: list[int]
    max_tokens: int = 8
    status: str = WAITING
    num_computed_tokens: int = 0
    n_output: int = 0

    @property
    def num_tokens(self) -> int:
        return len(self.tokens)


@dataclass
class NewRequestData:
    req_id: str
    token_ids: list[int]
    block_ids: list[int]
    num_computed_tokens: int


@dataclass
class SchedulerOutput:
    new_reqs: list[NewRequestData] = field(default_factory=list)
    block_deltas: dict[str, list[int]] = field(default_factory=dict)
    num_scheduled_tokens: dict[str, int] = field(default_factory=dict)
    finished_req_ids: set[str] = field(default_factory=set)
    preempted_req_ids: set[str] = field(default_factory=set)


# ======================================================================
# 3. scheduler -- identical to engine.py
# ======================================================================
class Scheduler:
    def __init__(self, kv: KVCacheManager, max_num_tokens: int = 8,
                 max_num_seqs: int = 4) -> None:
        self.kv = kv
        self.max_num_tokens = max_num_tokens
        self.max_num_seqs = max_num_seqs
        self.requests: dict[str, Request] = {}
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.finished_req_ids: set[str] = set()
        self.events: list[str] = []

    def add_request(self, req: Request) -> None:
        self.requests[req.req_id] = req
        self.waiting.append(req)

    def schedule(self) -> SchedulerOutput:
        self.events = []
        out = SchedulerOutput()
        budget = self.max_num_tokens

        scheduled_running: list[Request] = []
        scheduled_new: list[Request] = []
        deltas: dict[str, list[int]] = {}
        num_scheduled: dict[str, int] = {}

        i = 0
        while i < len(self.running) and budget > 0:
            req = self.running[i]
            n = min(req.num_tokens - req.num_computed_tokens, budget)
            if n <= 0:
                i += 1
                continue

            new_blocks = self.kv.allocate_slots(req, n)
            if new_blocks is None:
                victim = self.running[-1]
                if victim is req:
                    break
                self.running.pop()
                self._preempt(victim, out)
                continue

            deltas[req.req_id] = new_blocks
            num_scheduled[req.req_id] = n
            scheduled_running.append(req)
            budget -= n
            i += 1

        while self.waiting and budget > 0 and len(self.running) < self.max_num_seqs:
            req = self.waiting[0]
            if req.num_computed_tokens == 0:
                adopted, num_computed = self.kv.get_computed_blocks(req)
                if num_computed:
                    self.events.append(f"{req.req_id} prefix hit {num_computed}")
            else:
                adopted, num_computed = [], req.num_computed_tokens

            n = min(req.num_tokens - num_computed, budget)
            new_blocks = self.kv.allocate_slots(req, n, adopted)
            if new_blocks is None:
                self.events.append(f"{req.req_id} blocked (no memory)")
                break

            self.waiting.popleft()
            req.status = RUNNING
            req.num_computed_tokens = num_computed
            self.running.append(req)
            scheduled_new.append(req)
            num_scheduled[req.req_id] = n
            budget -= n

        for req in scheduled_running:
            if not deltas[req.req_id]:
                deltas.pop(req.req_id)

        out.new_reqs = [
            NewRequestData(
                req_id=r.req_id,
                token_ids=list(r.tokens),
                block_ids=self.kv.block_table(r),
                num_computed_tokens=r.num_computed_tokens,
            )
            for r in scheduled_new
        ]
        out.block_deltas = deltas
        out.num_scheduled_tokens = num_scheduled
        out.finished_req_ids = self.finished_req_ids

        self._update_after_schedule(out)
        return out

    def _update_after_schedule(self, out: SchedulerOutput) -> None:
        for req_id, n in out.num_scheduled_tokens.items():
            self.requests[req_id].num_computed_tokens += n
        self.finished_req_ids = set()

    def _preempt(self, req: Request, out: SchedulerOutput) -> None:
        self.kv.free(req)
        req.num_computed_tokens = 0
        req.status = PREEMPTED
        self.waiting.appendleft(req)
        out.preempted_req_ids.add(req.req_id)
        self.events.append(f"preempt {req.req_id}")

    def update_from_output(self, out: SchedulerOutput,
                           sampled: dict[str, list[int]]) -> None:
        for req_id in out.num_scheduled_tokens:
            req = self.requests[req_id]
            if req.num_computed_tokens < req.num_tokens:
                continue
            for token in sampled.get(req_id, ()):
                req.tokens.append(token)
                req.n_output += 1
            if req.n_output >= req.max_tokens:
                self._free_request(req)

    def _free_request(self, req: Request) -> None:
        if req in self.running:
            self.running.remove(req)
        self.kv.free(req)
        req.status = DONE
        self.requests.pop(req.req_id, None)
        self.finished_req_ids.add(req.req_id)
        self.events.append(f"{req.req_id} done")


# ======================================================================
# 4. model runner -- THIS is the new part. Real tensors.
# ======================================================================
@dataclass
class CachedRequestState:
    req_id: str
    token_ids: list[int]
    block_ids: list[int]
    num_computed_tokens: int


class GPUModelRunner:
    """Owns every GPU-side object. Lists in, tensors out."""

    def __init__(self, num_blocks: int, block_size: int,
                 max_num_tokens: int = 16) -> None:
        self.block_size = block_size

        # --- the KV cache: ONE tensor, laid out by block ---
        # vLLM: [num_blocks, block_size, 2, num_kv_heads, head_dim], per layer.
        self.kv_cache = torch.zeros(num_blocks, block_size, 2, HEAD_DIM)

        # --- persistent input buffers, allocated once at max size ---
        # vLLM keeps these in InputBatch and writes into slices each step.
        self.input_ids = torch.zeros(max_num_tokens, dtype=torch.int32)
        self.positions = torch.zeros(max_num_tokens, dtype=torch.int64)
        self.slot_mapping = torch.zeros(max_num_tokens, dtype=torch.int64)

        # --- a toy model: embedding + projections ---
        g = torch.Generator().manual_seed(0)
        self.embed = torch.randn(VOCAB, HEAD_DIM, generator=g)
        self.wq = torch.randn(HEAD_DIM, HEAD_DIM, generator=g) * 0.3
        self.wk = torch.randn(HEAD_DIM, HEAD_DIM, generator=g) * 0.3
        self.wv = torch.randn(HEAD_DIM, HEAD_DIM, generator=g) * 0.3
        self.lm_head = torch.randn(HEAD_DIM, VOCAB, generator=g)

        self.verify = True  # cross-check paged attention against a gather
        self.states: dict[str, CachedRequestState] = {}

    # -- the entry point ------------------------------------------------
    def execute(self, out: SchedulerOutput) -> dict[str, list[int]]:
        self._update_states(out)
        total = self._prepare_inputs(out)
        self._write_kv(total)
        return self._sample(out)

    # -- 1. apply the diff (identical logic to the CPU version) ---------
    def _update_states(self, out: SchedulerOutput) -> None:
        for req_id in out.finished_req_ids | out.preempted_req_ids:
            self.states.pop(req_id, None)
        for data in out.new_reqs:
            self.states[data.req_id] = CachedRequestState(
                req_id=data.req_id,
                token_ids=list(data.token_ids),
                block_ids=list(data.block_ids),
                num_computed_tokens=data.num_computed_tokens,
            )
        for req_id, new_blocks in out.block_deltas.items():
            self.states[req_id].block_ids.extend(new_blocks)

    # -- 2. lists -> tensors --------------------------------------------
    def _prepare_inputs(self, out: SchedulerOutput) -> int:
        """Write the flattened batch into the pre-allocated buffers."""
        offset = 0
        for req_id, n in out.num_scheduled_tokens.items():
            state = self.states[req_id]
            start = state.num_computed_tokens
            total = offset + n

            # input ids and positions for this request's slice
            self.input_ids[offset:total] = torch.tensor(
                state.token_ids[start : start + n], dtype=torch.int32
            )
            self.positions[offset:total] = torch.arange(start, start + n)

            # slot = block_id * block_size + offset_in_block
            for p in range(start, start + n):
                block_id = state.block_ids[p // self.block_size]
                self.slot_mapping[offset + (p - start)] = (
                    block_id * self.block_size + p % self.block_size
                )

            state.num_computed_tokens += n
            offset = total
        return offset

    # -- 3. one scatter writes the whole step's KV ----------------------
    def _write_kv(self, total: int) -> None:
        if total == 0:
            return
        tokens = self.input_ids[:total].long() % VOCAB
        k_new = self.embed[tokens] @ self.wk          # [T, D]
        v_new = self.embed[tokens] @ self.wv

        # THE line: scatter every token's K/V into its page.
        flat = self.kv_cache.view(-1, 2, HEAD_DIM)    # [num_slots, 2, D]
        flat[self.slot_mapping[:total], 0] = k_new
        flat[self.slot_mapping[:total], 1] = v_new

    # -- 4. paged attention: walk the block table, never gather ---------
    def _attend(self, state: CachedRequestState) -> torch.Tensor:
        seq_len = state.num_computed_tokens
        last_token = state.token_ids[seq_len - 1] % VOCAB
        q = self.embed[last_token] @ self.wq         # [D]

        acc = torch.zeros(HEAD_DIM)
        m = torch.tensor(float("-inf"))
        l = torch.tensor(0.0)

        num_blocks = -(-seq_len // self.block_size)
        for b in range(num_blocks):
            block_id = state.block_ids[b]
            k = self.kv_cache[block_id, :, 0]         # [block_size, D]
            v = self.kv_cache[block_id, :, 1]
            valid = min(self.block_size, seq_len - b * self.block_size)
            k, v = k[:valid], v[:valid]

            scores = (k @ q) / (HEAD_DIM ** 0.5)
            m_new = torch.maximum(m, scores.max())
            alpha = torch.exp(m - m_new)
            p = torch.exp(scores - m_new)
            l = l * alpha + p.sum()
            acc = acc * alpha + p @ v
            m = m_new

        out = acc / l

        if self.verify:
            # Proof that paging is only an addressing scheme: gather the same
            # history contiguously and run plain attention. Must match exactly.
            ids = torch.tensor(state.block_ids, dtype=torch.long)
            history = self.kv_cache[ids].reshape(-1, 2, HEAD_DIM)[:seq_len]
            k, v = history[:, 0], history[:, 1]
            scores = (k @ q) / (HEAD_DIM ** 0.5)
            reference = torch.softmax(scores, dim=-1) @ v
            assert torch.allclose(out, reference, atol=1e-5), (
                "paged attention disagrees with the gathered reference"
            )
        return out

    # -- 5. sample (greedy, only at the end of the sequence) ------------
    def _sample(self, out: SchedulerOutput) -> dict[str, list[int]]:
        sampled: dict[str, list[int]] = {}
        for req_id in out.num_scheduled_tokens:
            state = self.states[req_id]
            if state.num_computed_tokens != len(state.token_ids):
                continue  # prefill chunk, no token yet
            hidden = self._attend(state)
            token = int((hidden @ self.lm_head).argmax())
            state.token_ids.append(token)
            sampled[req_id] = [token]
        return sampled


# ======================================================================
# 5. drive it
# ======================================================================
def main() -> None:
    torch.manual_seed(0)
    num_blocks, block_size = 6, 4          # 24 KV slots
    kv = KVCacheManager(num_blocks, block_size)
    sched = Scheduler(kv, max_num_tokens=8, max_num_seqs=4)
    runner = GPUModelRunner(num_blocks, block_size, max_num_tokens=8)

    sched.add_request(Request("r0", [1, 2, 3, 4, 5, 0], max_tokens=3))
    sched.add_request(Request("r1", [2, 3, 4, 5, 6, 7, 1, 2, 3, 4], max_tokens=2))
    sched.add_request(Request("r2", [4, 5, 6, 7], max_tokens=2))

    print("r0: 6 tokens, r1: 10 tokens, r2: 4 tokens, budget 8/step,"
          f" {num_blocks * block_size} KV slots")
    step = 0
    while sched.waiting or sched.running:
        step += 1
        out = sched.schedule()
        sampled = runner.execute(out)
        print("sampled: ", sampled)
        plan = " ".join(f"{r}:{n}" for r, n in out.num_scheduled_tokens.items()) or "-"
        total = sum(out.num_scheduled_tokens.values())
        print(f"  step {step:2d} | {plan:<14} | {total} tok "
              f"running={[r.req_id for r in sched.running]} "
              f"free={kv.num_free_blocks}")
        for event in sched.events:
            print(f"           |   {event}")
        sched.update_from_output(out, sampled)

    assert not sched.waiting and not sched.running
    assert kv.num_free_blocks == num_blocks, "block leak"
    print(f"  done. all {num_blocks} blocks returned.")

    # What the KV cache actually looks like now: one tensor, paged.
    print(f"\n  kv_cache shape {tuple(runner.kv_cache.shape)}"
          f"  ({num_blocks} blocks x {block_size} slots x 2 (K,V) x {HEAD_DIM})")
    written = (runner.kv_cache.abs().sum(dim=(2, 3)) > 0).sum().item()
    print(f"  slots ever written: {written}")


if __name__ == "__main__":
    main()
