"""
scheduler_vllm_style.py

The same scheduler, but with vLLM's actual architecture.
Read after scheduler.py, which taught the logic. This file teaches the shape.

Three decisions make vLLM's version look different
---------------------------------------------------
1. COLUMNAR, NOT PER-REQUEST.
   The plan is `dict[req_id, num_tokens]` plus parallel lists. There is no
   `PlanItem`. One Python object per request per step is too expensive at
   thousands of steps/s, and a graph of objects does not serialize over ZMQ.

2. BLOCK IDS TRAVEL AS A DIFF.
   `NewRequestData` carries the full block table, because the worker has
   never seen that request. `CachedRequestData` carries only the blocks
   allocated THIS step; the worker appends them to its own copy. Sending a
   100k-token request's 6000 block ids every step would dwarf the work.

3. TWO COPIES OF STATE, SYNCED BY TWO NUMBERS.
   The scheduler owns `Request`. The worker owns `CachedRequestState`.
   Per step they exchange `num_computed_tokens` and `num_scheduled_tokens`.
   Positions, slot mapping, seq lengths: all derived on the worker side.

The ordering rule that trips people up
--------------------------------------
`num_computed_tokens` is advanced in `_update_after_schedule`, AFTER the
output is built -- because the worker needs the PRE-step value to know which
positions this step covers. Build the output with the old value, then advance.

What is still faked
-------------------
No KV connectors, no encoder cache, no spec decode, no async scheduling, no
priority policy. Every one of those attaches at a single point in `schedule()`
and none of them change the three decisions above.
"""

from collections import deque
from dataclasses import dataclass, field

import torch


# ======================================================================
# 1. state
# ======================================================================
class Block:
    __slots__ = ("block_id", "ref_count", "hash")

    def __init__(self, block_id: int) -> None:
        self.block_id = block_id
        self.ref_count = 0
        self.hash: int | None = None


WAITING, RUNNING, PREEMPTED, FINISHED = "waiting", "running", "preempted", "finished"


@dataclass
class Request:
    """The scheduler's copy. The worker has its own (CachedRequestState)."""

    req_id: str
    tokens: list[int]
    max_tokens: int = 8
    status: str = WAITING
    num_computed_tokens: int = 0
    n_output: int = 0

    @property
    def num_tokens(self) -> int:
        return len(self.tokens)

    @property
    def is_finished(self) -> bool:
        return self.status == FINISHED


# ======================================================================
# 2. KV memory -- returns the blocks it allocated, nothing else
# ======================================================================
class KVCacheManager:
    """Owns the block pool and one block table per request.

    The important API change from kv_cache_core.py: `allocate_slots` returns
    only the NEWLY allocated block ids. The scheduler forwards exactly those
    to the worker, which appends them. That is the diff.
    """

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
        if block.hash is not None:  # evicted; its KV is now garbage
            del self.cached[block.hash]
            block.hash = None
        block.ref_count = 1
        return block.block_id

    def _release(self, block_id: int) -> None:
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_blocks.append(block_id)

    # -- prefix cache --------------------------------------------------
    def get_computed_blocks(self, request: Request) -> tuple[list[int], int]:
        """Longest cached prefix. Takes a reference on each hit.

        Returns (block_ids, num_tokens) -- the caller must hand these to
        `allocate_slots`, or release them if it gives up.
        """
        blocks: list[int] = []
        parent = None
        n = 0
        # `- 1`: the block holding the last token is never reusable, we need
        # one token left to run the model on.
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

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        new_computed_blocks: list[int] | None = None,
    ) -> list[int] | None:
        """Reserve blocks. Returns the ids allocated this call, or None.

        `new_computed_blocks` are prefix-cache hits already referenced by
        `get_computed_blocks`; they count toward this request's table and are
        part of what we return.
        """
        adopted = new_computed_blocks or []
        num_computed = request.num_computed_tokens + len(adopted) * self.block_size

        # Blocks needed for the WHOLE sequence, minus the ones this request
        # already holds, minus the cached ones we are adopting.
        required = -(-(num_computed + num_new_tokens) // self.block_size)
        existing = len(self.req_to_blocks.get(request.req_id, []))
        num_new = required - existing - len(adopted)

        if num_new > self.num_free_blocks:
            # Atomic failure: undo the prefix adoption so nothing leaks.
            for block_id in adopted:
                self._release(block_id)
            return None

        allocated = list(adopted)
        for _ in range(num_new):
            allocated.append(self._take())

        self.req_to_blocks.setdefault(request.req_id, []).extend(allocated)
        self._cache_full_blocks(request, num_computed)
        return allocated

    def _cache_full_blocks(self, request: Request, up_to: int) -> None:
        """Publish blocks that are fully computed, so others can reuse them."""
        table = self.req_to_blocks[request.req_id]
        parent = None
        for i, block_id in enumerate(table):
            end = (i + 1) * self.block_size
            if end > up_to:
                break
            key = hash((parent, tuple(request.tokens[end - self.block_size : end])))
            block = self.blocks[block_id]
            if block.hash is None and key not in self.cached:
                block.hash = key
                self.cached[key] = block
            parent = key

    def free(self, request: Request) -> None:
        for block_id in self.req_to_blocks.pop(request.req_id, []):
            self._release(block_id)

    def get_block_ids(self, req_id: str) -> tuple[list[int], ...]:
        # A tuple of lists: one per KV cache group. Hybrid models (attention
        # + mamba) genuinely have several; we only ever have one.
        return (self.req_to_blocks.get(req_id, []),)


# ======================================================================
# 3. the wire format -- this is what the scheduler sends
# ======================================================================
@dataclass
class NewRequestData:
    """Full payload. Sent once, the step a request is first scheduled."""

    req_id: str
    prompt_token_ids: list[int]
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int


@dataclass
class CachedRequestData:
    """Parallel lists, one entry per request already known to the worker."""

    req_ids: list[str] = field(default_factory=list)
    # For these, new_block_ids[i] REPLACES the block table instead of appending.
    resumed_req_ids: set[str] = field(default_factory=set)
    new_block_ids: list[tuple[list[int], ...] | None] = field(default_factory=list)
    num_computed_tokens: list[int] = field(default_factory=list)
    num_output_tokens: list[int] = field(default_factory=list)

    @property
    def num_reqs(self) -> int:
        return len(self.req_ids)


@dataclass
class SchedulerOutput:
    scheduled_new_reqs: list[NewRequestData] = field(default_factory=list)
    scheduled_cached_reqs: CachedRequestData = field(default_factory=CachedRequestData)
    num_scheduled_tokens: dict[str, int] = field(default_factory=dict)
    total_num_scheduled_tokens: int = 0
    finished_req_ids: set[str] = field(default_factory=set)
    preempted_req_ids: set[str] = field(default_factory=set)


# ======================================================================
# 4. the scheduler
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
        self.finished_req_ids: set[str] = set()   # flushed each step
        self.events: list[str] = []

    def add_request(self, req: Request) -> None:
        self.requests[req.req_id] = req
        self.waiting.append(req)

    # ------------------------------------------------------------------
    def schedule(self) -> SchedulerOutput:
        self.events = []
        out = SchedulerOutput()
        budget = self.max_num_tokens

        scheduled_running: list[Request] = []
        scheduled_resumed: list[Request] = []
        scheduled_new: list[Request] = []
        req_to_new_blocks: dict[str, list[int]] = {}
        num_scheduled: dict[str, int] = {}

        # --- a. running requests ---
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
                self._preempt_request(victim, out)
                continue

            req_to_new_blocks[req.req_id] = new_blocks
            num_scheduled[req.req_id] = n
            scheduled_running.append(req)
            budget -= n
            i += 1

        # --- b. waiting requests ---
        while self.waiting and budget > 0 and len(self.running) < self.max_num_seqs:
            req = self.waiting[0]

            if req.num_computed_tokens == 0:
                # Fresh, or preempted and reset: either way, look for a
                # prefix someone else already computed.
                blocks, num_computed = self.kv.get_computed_blocks(req)
                if num_computed:
                    self.events.append(f"{req.req_id} prefix hit {num_computed}")
            else:
                blocks, num_computed = [], req.num_computed_tokens

            n = min(req.num_tokens - num_computed, budget)
            new_blocks = self.kv.allocate_slots(req, n, blocks)
            if new_blocks is None:
                self.events.append(f"{req.req_id} blocked (no memory)")
                break
            self.waiting.popleft()

            was_preempted = req.status == PREEMPTED
            req.status = RUNNING
            req.num_computed_tokens = num_computed
            self.running.append(req)

            req_to_new_blocks[req.req_id] = new_blocks
            num_scheduled[req.req_id] = n
            (scheduled_resumed if was_preempted else scheduled_new).append(req)
            budget -= n

        # --- c. build the output (PRE-step num_computed_tokens) ---
        out.scheduled_new_reqs = [
            NewRequestData(
                req_id=r.req_id,
                prompt_token_ids=r.tokens,
                block_ids=self.kv.get_block_ids(r.req_id),
                num_computed_tokens=r.num_computed_tokens,
            )
            for r in scheduled_new
        ]

        cached = CachedRequestData()
        for req in scheduled_running + scheduled_resumed:
            cached.req_ids.append(req.req_id)
            cached.new_block_ids.append(
                (req_to_new_blocks[req.req_id],) if req_to_new_blocks[req.req_id] else None
            )
            cached.num_computed_tokens.append(req.num_computed_tokens)
            cached.num_output_tokens.append(req.n_output)
        cached.resumed_req_ids = {r.req_id for r in scheduled_resumed}
        out.scheduled_cached_reqs = cached

        out.num_scheduled_tokens = num_scheduled
        out.total_num_scheduled_tokens = sum(num_scheduled.values())
        out.finished_req_ids = self.finished_req_ids

        self._update_after_schedule(out)
        return out

    def _update_after_schedule(self, out: SchedulerOutput) -> None:
        """Advance counters AFTER the output is built. Order matters."""
        for req_id, n in out.num_scheduled_tokens.items():
            self.requests[req_id].num_computed_tokens += n
        self.finished_req_ids = set()

    def _preempt_request(self, req: Request, out: SchedulerOutput) -> None:
        self.kv.free(req)                 # gives every block back
        req.num_computed_tokens = 0       # it must recompute from scratch
        req.status = PREEMPTED
        self.waiting.appendleft(req)
        out.preempted_req_ids.add(req.req_id)
        self.events.append(f"preempt {req.req_id}")

    # ------------------------------------------------------------------
    def update_from_output(self, out: SchedulerOutput,
                           model_output: dict[str, list[int]]) -> None:
        for req_id, n in out.num_scheduled_tokens.items():
            req = self.requests[req_id]
            # The scheduler has already advanced the counter, so this step
            # reached the end of the sequence only if computed == num_tokens.
            if req.num_computed_tokens < req.num_tokens:
                continue
            for token in model_output.get(req_id, ()):
                req.tokens.append(token)
                req.n_output += 1
            if req.n_output >= req.max_tokens:
                self._free_request(req)

    def _free_request(self, req: Request) -> None:
        if req in self.running:
            self.running.remove(req)
        self.kv.free(req)
        req.status = FINISHED
        self.requests.pop(req.req_id, None)
        self.finished_req_ids.add(req.req_id)
        self.events.append(f"{req.req_id} done")

    # ------------------------------------------------------------------
    def run(self) -> None:
        # ONE runner: it holds the worker's copy of every request.
        runner = ModelRunner(self.kv.block_size)
        step = 0
        while self.waiting or self.running:
            step += 1
            out = self.schedule()
            batch = runner.execute(out)
            plan = " ".join(f"{r}:{n}" for r, n in out.num_scheduled_tokens.items()) or "-"
            print(f"  step {step:2d} | {plan:<14} | tokens={len(batch.input_ids)} "
                  f"running={[r.req_id for r in self.running]} "
                  f"free={self.kv.num_free_blocks}")
            for e in self.events:
                print(f"           |   {e}")
            self.update_from_output(out, batch.sampled)


# ======================================================================
# 5. the worker side -- applies the diff
# ======================================================================
@dataclass
class CachedRequestState:
    req_id: str
    token_ids: list[int]
    block_ids: list[int]
    num_computed_tokens: int


@dataclass
class Batch:
    input_ids: torch.Tensor
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    seq_lens: torch.Tensor
    sampled: dict[str, list[int]] = field(default_factory=dict)


class ModelRunner:
    """Stand-in for GPUModelRunner: owns state, applies diffs, flattens."""

    def __init__(self, block_size: int) -> None:
        self.block_size = block_size
        self.requests: dict[str, CachedRequestState] = {}

    def execute(self, out: SchedulerOutput) -> Batch:
        self._update_states(out)
        return self._prepare_inputs(out)

    def _update_states(self, out: SchedulerOutput) -> None:
        for req_id in out.finished_req_ids:
            self.requests.pop(req_id, None)

        # Preempted: the KV is gone, but keep the shell so the block table can
        # be replaced when it resumes.
        for req_id in out.preempted_req_ids:
            state = self.requests.get(req_id)
            if state is not None:
                state.block_ids = []
                state.num_computed_tokens = 0

        for data in out.scheduled_new_reqs:
            self.requests[data.req_id] = CachedRequestState(
                req_id=data.req_id,
                token_ids=list(data.prompt_token_ids),
                block_ids=list(data.block_ids[0]),   # FULL table
                num_computed_tokens=data.num_computed_tokens,
            )

        cached = out.scheduled_cached_reqs
        for i, req_id in enumerate(cached.req_ids):
            state = self.requests[req_id]
            new_blocks = cached.new_block_ids[i]
            if new_blocks is not None:
                if req_id in cached.resumed_req_ids:
                    state.block_ids = list(new_blocks[0])   # replace
                else:
                    state.block_ids.extend(new_blocks[0])   # append the diff
            state.num_computed_tokens = cached.num_computed_tokens[i]

    def _prepare_inputs(self, out: SchedulerOutput) -> Batch:
        """Flatten the columnar plan into packed tensors (vLLM uses np.repeat)."""
        ids: list[int] = []
        positions: list[int] = []
        slots: list[int] = []
        seq_lens: list[int] = []
        sampled: dict[str, list[int]] = {}

        for req_id, n in out.num_scheduled_tokens.items():
            state = self.requests[req_id]
            start = state.num_computed_tokens

            ids.extend(state.token_ids[start : start + n])
            positions.extend(range(start, start + n))
            for p in range(start, start + n):
                block = state.block_ids[p // self.block_size]
                slots.append(block * self.block_size + p % self.block_size)

            state.num_computed_tokens += n
            seq_lens.append(state.num_computed_tokens)

            # A step produces a token only if it consumed the whole sequence.
            if state.num_computed_tokens == len(state.token_ids):
                token = 100 + len(state.token_ids)
                state.token_ids.append(token)
                sampled[req_id] = [token]

        return Batch(
            input_ids=torch.tensor(ids, dtype=torch.long),
            positions=torch.tensor(positions, dtype=torch.long),
            slot_mapping=torch.tensor(slots, dtype=torch.long),
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
            sampled=sampled,
        )


# ======================================================================
# demo: same scenario as scheduler.py, different architecture
# ======================================================================
if __name__ == "__main__":
    sched = Scheduler(KVCacheManager(num_blocks=6, block_size=4),
                      max_num_tokens=8, max_num_seqs=4)

    sched.add_request(Request("r0", list(range(6)), max_tokens=2))
    sched.add_request(Request("r1", list(range(20)), max_tokens=2))
    sched.add_request(Request("r2", list(range(4)), max_tokens=2))

    print("r0: 6 tokens, r1: 20 tokens, r2: 4 tokens, budget 8/step")
    sched.run()

    assert not sched.waiting and not sched.running
    assert sched.kv.num_free_blocks == 6, "block leak"
    print("  all requests done, all 6 blocks returned")
