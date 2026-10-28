"""
engine.py  --  THE ONE FILE.  Everything else in this folder is scratch.

A mini vLLM engine in one file: KV allocator, scheduler, and model runner.
~300 lines. This is the version to memorize and to write in an interview.

=====================================================================
THE SIX OBJECTS AND WHO OWNS WHAT
=====================================================================
    Block            one page of KV memory: id, ref_count, hash
    KVCacheManager   all blocks + prefix cache + one block table per request
    Request          the SCHEDULER's per-request state
    SchedulerOutput  the plan, sent scheduler -> runner
    Scheduler        decides who runs and how much
    ModelRunner      the WORKER's per-request state + turns the plan into work

Two objects hold the same request, and that is deliberate:

    Request           (scheduler side)   tokens, num_computed_tokens, status
    CachedRequestState (runner side)     token_ids, block_ids, num_computed_tokens

They never share a pointer. They exchange two numbers per step:
`num_scheduled_tokens` going down, `num_computed_tokens` going up.

=====================================================================
THE TWO INVARIANTS
=====================================================================
1.  block.ref_count == 0  <=>  block_id is in the free queue
    A block with ref_count 0 but a hash is BOTH free and cached. That is the
    whole trick of prefix caching, and every bug is a violation of this.

2.  sum(ref_count) == sum(len(block_table)) over all live requests
    Blocks are conserved. If they are not, you have a leak.

=====================================================================
THE THREE DECISIONS THAT MATTER
=====================================================================
1.  Block ids travel as a DIFF. `allocate_slots` returns only the NEWLY
    allocated blocks. The runner appends them to its own table. Sending a
    100k-token request's whole table every step would dwarf the actual work.

2.  `num_computed_tokens` is advanced AFTER the output is built
    (`_update_after_schedule`). The runner needs the PRE-step value to know
    which positions this step covers.

3.  Preemption is for RUNNING requests only. A waiting request that does not
    fit is simply not admitted this step. Otherwise you livelock: admit A,
    preempt for A, admit B, preempt A for B, forever.

=====================================================================
WHAT TO SAY OUT LOUD
=====================================================================
  * "One budget is tokens per step, the other is free blocks. The scheduler
     hands out slices of both, in order, and the loser gets preempted."
  * "There is no prefill phase and no decode phase. Every request is just
     `num_computed_tokens` catching up to `num_tokens`. Chunked prefill,
     prefix caching and speculative decoding all fall out of that."
  * "Preemption is not cancel. It is rewind to zero, give the blocks back,
     and go to the front of the queue."
  * "The scheduler never touches a tensor. It produces a plan; the runner
     replays it."

Deliberately missing: torch, CUDA graphs, KV connectors, spec decode, LoRA,
multimodal, priority policy. Each attaches at ONE point in `schedule()` and
none of them change the shape of the loop.
"""

from collections import deque
from dataclasses import dataclass, field


# ======================================================================
# 1. memory
# ======================================================================
class Block:
    """One page: block_size token positions of KV."""

    __slots__ = ("block_id", "ref_count", "hash")

    def __init__(self, block_id: int) -> None:
        self.block_id = block_id
        self.ref_count = 0
        self.hash: int | None = None


class KVCacheManager:
    """Owns every block, the prefix cache, and one block table per request."""

    def __init__(self, num_blocks: int, block_size: int) -> None:
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.free_blocks: deque[int] = deque(range(num_blocks))  # left = evicted first
        self.cached: dict[int, Block] = {}                # hash -> block
        self.req_to_blocks: dict[str, list[int]] = {}

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    # -- internals ------------------------------------------------------
    def _take(self) -> int:
        """Hand out one block. Anything it cached is now gone (eviction)."""
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
            self.free_blocks.append(block_id)  # keeps its hash: still reusable

    # -- prefix cache ---------------------------------------------------
    def get_computed_blocks(self, request) -> tuple[list[int], int]:
        """Longest already-cached prefix. Takes a reference on each hit.

        Returns (block_ids, tokens_reused). The caller passes those ids to
        `allocate_slots`, or releases them if it gives up.
        """
        blocks: list[int] = []
        parent = None
        n = 0
        # `- 1`: never take the block holding the LAST token. We need one
        # token left to run the model on, or there are no logits to sample.
        while n + self.block_size <= len(request.tokens) - 1:
            key = hash((parent, tuple(request.tokens[n : n + self.block_size])))
            block = self.cached.get(key)
            if block is None:
                break
            if block.ref_count == 0:
                self.free_blocks.remove(block.block_id)  # wake it up
            block.ref_count += 1
            blocks.append(block.block_id)
            parent = key
            n += self.block_size
        return blocks, n

    # -- the one call the scheduler makes -------------------------------
    def allocate_slots(self, request, num_new_tokens: int,
                       adopted: list[int] | None = None) -> list[int] | None:
        """Reserve blocks. Returns the ids allocated THIS CALL, or None.

        None means nothing changed, so the scheduler can preempt and retry.
        """
        adopted = adopted or []
        num_computed = request.num_computed_tokens + len(adopted) * self.block_size

        required = -(-(num_computed + num_new_tokens) // self.block_size)
        existing = len(self.req_to_blocks.get(request.req_id, []))
        num_new = required - existing - len(adopted)

        if num_new > self.num_free_blocks:
            for block_id in adopted:  # undo the prefix adoption
                self._release(block_id)
            return None

        allocated = list(adopted)
        for _ in range(num_new):
            allocated.append(self._take())
        self.req_to_blocks.setdefault(request.req_id, []).extend(allocated)
        self._cache_full_blocks(request, num_computed)
        return allocated

    def _cache_full_blocks(self, request, up_to: int) -> None:
        """Publish blocks that are fully computed, so others can reuse them."""
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
# 2. request + plan
# ======================================================================
WAITING, RUNNING, PREEMPTED, FINISHED = "waiting", "running", "preempted", "done"


@dataclass
class Request:
    req_id: str
    tokens: list[int]              # prompt + generated so far
    max_tokens: int = 8
    status: str = WAITING
    num_computed_tokens: int = 0   # how much KV actually exists
    n_output: int = 0

    @property
    def num_tokens(self) -> int:
        return len(self.tokens)


@dataclass
class NewRequestData:
    """Sent once: the runner has no state for this request yet."""

    req_id: str
    token_ids: list[int]
    block_ids: list[int]           # the FULL table
    num_computed_tokens: int


@dataclass
class SchedulerOutput:
    new_reqs: list[NewRequestData] = field(default_factory=list)
    block_deltas: dict[str, list[int]] = field(default_factory=dict)  # APPEND these
    num_scheduled_tokens: dict[str, int] = field(default_factory=dict)
    finished_req_ids: set[str] = field(default_factory=set)
    preempted_req_ids: set[str] = field(default_factory=set)


# ======================================================================
# 3. scheduler
# ======================================================================
class Scheduler:
    def __init__(self, kv: KVCacheManager, max_num_tokens: int = 8,
                 max_num_seqs: int = 4) -> None:
        self.kv = kv
        self.max_num_tokens = max_num_tokens  # token budget per step
        self.max_num_seqs = max_num_seqs      # requests per batch
        self.requests: dict[str, Request] = {}
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.finished_req_ids: set[str] = set()
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
        scheduled_new: list[Request] = []
        deltas: dict[str, list[int]] = {}
        num_scheduled: dict[str, int] = {}

        # --- a. running requests first: they are already owed work ---
        i = 0
        while i < len(self.running) and budget > 0:
            req = self.running[i]
            n = min(req.num_tokens - req.num_computed_tokens, budget)
            if n <= 0:
                i += 1
                continue

            new_blocks = self.kv.allocate_slots(req, n)
            if new_blocks is None:
                # No memory: kill the NEWEST running request and retry.
                victim = self.running[-1]
                if victim is req:
                    break  # nothing left to kill
                self.running.pop()
                self._preempt(victim, out)
                continue

            deltas[req.req_id] = new_blocks
            num_scheduled[req.req_id] = n
            scheduled_running.append(req)
            budget -= n
            i += 1

        # --- b. admit waiting requests, never at a running one's expense ---
        while self.waiting and budget > 0 and len(self.running) < self.max_num_seqs:
            req = self.waiting[0]

            if req.num_computed_tokens == 0:
                # Fresh, or preempted-and-reset: look for a cached prefix.
                adopted, num_computed = self.kv.get_computed_blocks(req)
                if num_computed:
                    self.events.append(f"{req.req_id} prefix hit {num_computed}")
            else:
                adopted, num_computed = [], req.num_computed_tokens

            n = min(req.num_tokens - num_computed, budget)
            new_blocks = self.kv.allocate_slots(req, n, adopted)
            if new_blocks is None:
                self.events.append(f"{req.req_id} blocked (no memory)")
                break  # <-- break, never preempt. See decision 3.

            self.waiting.popleft()
            req.status = RUNNING
            req.num_computed_tokens = num_computed
            self.running.append(req)
            scheduled_new.append(req)
            num_scheduled[req.req_id] = n
            budget -= n

        # --- c. build the plan with PRE-step counters ---
        for req in scheduled_running:
            if not deltas[req.req_id]:
                deltas.pop(req.req_id)  # nothing new: no delta to send

        # A request is "new" to the runner if the runner has no state for it.
        # Fresh admissions and resumes both qualify (we drop state on preempt).
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
        """Advance counters AFTER the plan is built. Order matters."""
        for req_id, n in out.num_scheduled_tokens.items():
            self.requests[req_id].num_computed_tokens += n
        self.finished_req_ids = set()

    def _preempt(self, req: Request, out: SchedulerOutput) -> None:
        """Not a cancel. Give the memory back and requeue at the front."""
        self.kv.free(req)
        req.num_computed_tokens = 0
        req.status = PREEMPTED
        self.waiting.appendleft(req)
        out.preempted_req_ids.add(req.req_id)
        self.events.append(f"preempt {req.req_id}")

    # ------------------------------------------------------------------
    def update_from_output(self, out: SchedulerOutput,
                           sampled: dict[str, list[int]]) -> None:
        for req_id in out.num_scheduled_tokens:
            req = self.requests[req_id]
            # Counters are already advanced, so the step reached the end of
            # the sequence only if computed == num_tokens.
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
        req.status = FINISHED
        self.requests.pop(req.req_id, None)
        self.finished_req_ids.add(req.req_id)
        self.events.append(f"{req.req_id} done")


# ======================================================================
# 4. model runner (the worker side)
# ======================================================================
@dataclass
class CachedRequestState:
    req_id: str
    token_ids: list[int]
    block_ids: list[int]
    num_computed_tokens: int


@dataclass
class Batch:
    """What a real runner would hand to the GPU kernel."""

    input_ids: list[int]
    positions: list[int]
    slot_mapping: list[int]   # block_id * block_size + offset, per token
    sampled: dict[str, list[int]] = field(default_factory=dict)


class ModelRunner:
    def __init__(self, block_size: int) -> None:
        self.block_size = block_size
        self.states: dict[str, CachedRequestState] = {}

    def execute(self, out: SchedulerOutput) -> Batch:
        self._update_states(out)
        return self._prepare_inputs(out)

    def _update_states(self, out: SchedulerOutput) -> None:
        # Order matters: drop first, then add. A request preempted in the
        # running loop can be re-admitted in the waiting loop of the same step.
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
            self.states[req_id].block_ids.extend(new_blocks)  # the diff

    def _prepare_inputs(self, out: SchedulerOutput) -> Batch:
        ids: list[int] = []
        positions: list[int] = []
        slots: list[int] = []
        sampled: dict[str, list[int]] = {}

        for req_id, n in out.num_scheduled_tokens.items():
            state = self.states[req_id]
            start = state.num_computed_tokens

            ids.extend(state.token_ids[start : start + n])
            positions.extend(range(start, start + n))
            for p in range(start, start + n):
                block_id = state.block_ids[p // self.block_size]
                slots.append(block_id * self.block_size + p % self.block_size)

            state.num_computed_tokens += n
            if state.num_computed_tokens == len(state.token_ids):
                token = 100 + len(state.token_ids)  # stand-in for the sampler
                state.token_ids.append(token)
                sampled[req_id] = [token]

        return Batch(ids, positions, slots, sampled)


# ======================================================================
# 5. drive it
# ======================================================================
def main() -> None:
    kv = KVCacheManager(num_blocks=6, block_size=4)   # 24 token slots
    sched = Scheduler(kv, max_num_tokens=8, max_num_seqs=4)
    runner = ModelRunner(block_size=4)

    sched.add_request(Request("r0", list(range(6)), max_tokens=2))
    sched.add_request(Request("r1", list(range(20)), max_tokens=2))
    sched.add_request(Request("r2", list(range(4)), max_tokens=2))

    print("r0: 6 tokens, r1: 20 tokens, r2: 4 tokens, budget 8/step")
    step = 0
    while sched.waiting or sched.running:
        step += 1
        out = sched.schedule()
        batch = runner.execute(out)
        plan = " ".join(f"{r}:{n}" for r, n in out.num_scheduled_tokens.items()) or "-"
        print(f"  step {step:2d} | {plan:<14} | tokens={len(batch.input_ids)} "
              f"running={[r.req_id for r in sched.running]} "
              f"free={kv.num_free_blocks}")
        for event in sched.events:
            print(f"           |   {event}")
        sched.update_from_output(out, batch.sampled)

    assert not sched.waiting and not sched.running
    assert kv.num_free_blocks == 6, "block leak"
    print("  all requests done, all 6 blocks returned")


if __name__ == "__main__":
    main()
