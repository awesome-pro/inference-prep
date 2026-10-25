"""
scheduler.py

What a scheduler actually does
------------------------------
It answers three questions every step, and nothing else:

  1. WHICH requests run this step?
  2. HOW MANY tokens of each?        (the token budget)
  3. Can we afford the KV memory?    (the block budget)

Then it hands a plan to the model. It never touches tensors.

The loop
--------
    plan  = scheduler.step()          # decide
    out   = model.run(plan)           # GPU
    scheduler.update(plan, out)       # settle: sample, stop, free

Two budgets, and that is the whole tension:

    token budget : max_num_tokens per step (compute)
    block budget : free blocks in the KV pool (memory)

The one rule that makes it stable
---------------------------------
  * A RUNNING request may preempt another running request to get memory.
  * A WAITING request may never preempt. If it does not fit, we simply stop
    admitting and try again next step.

Break that rule and you get livelock: admit A, preempt someone for A, next
step admit B, preempt A for B, forever. New work must wait for old work.

The core idea (same as vLLM)
----------------------------
There is no "prefill phase" and no "decode phase". Every request is just

    computed  --->  len(tokens)

and each step we hand out tokens to close that gap. Big gap = prefill.
Gap of 1 = decode. A prefill split across steps = chunked prefill.
"""

from collections import deque
from dataclasses import dataclass, field


# ======================================================================
# KV memory (the short version of kv_cache_core.py)
# ======================================================================
class Block:
    """One page of KV memory: block_size token positions."""

    __slots__ = ("block_id", "ref_count", "hash")

    def __init__(self, block_id: int) -> None:
        self.block_id = block_id
        self.ref_count = 0          # how many requests are using it
        self.hash: int | None = None  # set once its tokens are cached


@dataclass
class Request:
    req_id: str
    tokens: list[int]              # prompt + everything generated
    max_tokens: int = 8

    table: list[Block] = field(default_factory=list)
    computed: int = 0              # how many positions have real KV
    n_output: int = 0              # how many tokens we generated

    @property
    def num_tokens(self) -> int:
        return len(self.tokens)

    def __repr__(self) -> str:
        return f"<{self.req_id} {self.computed}/{self.num_tokens}>"


class KVStore:
    """Every block, plus the prefix cache.

    The scheduler calls four things:
        alloc(req, n) -> bool    reserve room for n more tokens
        reuse(req)               adopt a cached prefix, returns tokens reused
        publish(req)             make full blocks reusable by others
        drop(req)                give every block back
    """

    def __init__(self, num_blocks: int, block_size: int) -> None:
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.free: deque[int] = deque(range(num_blocks))  # left = evict first
        self.cached: dict[int, Block] = {}                # hash -> block

    # -- the only decision the scheduler asks about --------------------
    def alloc(self, req: Request, n: int) -> bool:
        """Reserve blocks. False means nothing changed (so retry is safe)."""
        need = -(-(req.computed + n) // self.block_size) - len(req.table)
        if need > len(self.free):
            return False

        for _ in range(need):
            block = self.blocks[self.free.popleft()]
            if block.hash is not None:      # eviction: its content is lost
                del self.cached[block.hash]
                block.hash = None
            block.ref_count = 1
            req.table.append(block)
        req.computed += n
        return True

    def drop(self, req: Request) -> None:
        """Give every block back. The request keeps its tokens, so a
        preempted request can come back later and recompute."""
        for block in req.table:
            block.ref_count -= 1
            if block.ref_count == 0:
                self.free.append(block.block_id)
        req.table.clear()
        req.computed = 0

    # -- prefix cache ---------------------------------------------------
    def reuse(self, req: Request) -> int:
        parent = None
        reused = 0
        # The `- 1`: never take the block holding the LAST token. We need one
        # token left to run the model on, or there are no logits to sample.
        while reused + self.block_size <= len(req.tokens) - 1:
            chunk = tuple(req.tokens[reused : reused + self.block_size])
            key = hash((parent, chunk))
            block = self.cached.get(key)
            if block is None:
                break
            if block.ref_count == 0:
                self.free.remove(block.block_id)
            block.ref_count += 1
            req.table.append(block)
            parent = key
            reused += self.block_size
        req.computed = reused
        return reused

    def publish(self, req: Request) -> None:
        """After the model runs: every block that is now full can be shared."""
        parent = None
        for i, block in enumerate(req.table):
            end = (i + 1) * self.block_size
            if end > req.computed:
                break
            chunk = tuple(req.tokens[end - self.block_size : end])
            key = hash((parent, chunk))
            if block.hash is None and key not in self.cached:
                block.hash = key
                self.cached[key] = block
            parent = key  # keep the chain going even on a cache hit


# ======================================================================
# the scheduler
# ======================================================================
@dataclass
class PlanItem:
    """One request's share of this step. This is what the model reads."""

    req: Request
    start: int            # first token position computed this step
    num_tokens: int       # how many positions
    block_ids: list[int]  # where their KV lives


class Scheduler:
    def __init__(self, kv: KVStore, max_num_tokens: int = 8,
                 max_num_seqs: int = 4) -> None:
        self.kv = kv
        self.max_num_tokens = max_num_tokens  # token budget per step
        self.max_num_seqs = max_num_seqs      # requests per batch
        self.waiting: deque[Request] = deque()  # never ran, or preempted
        self.running: list[Request] = []        # admitted, owed work
        self.events: list[str] = []             # why decisions were made

    def add_request(self, req: Request) -> None:
        self.waiting.append(req)

    # ------------------------------------------------------------------
    # 1. decide
    # ------------------------------------------------------------------
    def step(self) -> list[PlanItem]:
        """Return the plan for this step. Touches no tensors."""
        self.events = []
        budget = self.max_num_tokens
        items: list[PlanItem] = []

        # --- a. running requests first: they are already owed work ---
        i = 0
        while i < len(self.running) and budget > 0:
            req = self.running[i]
            n = min(req.num_tokens - req.computed, budget)
            if n <= 0:
                i += 1
                continue

            if not self.kv.alloc(req, n):
                # No memory. Kill the NEWEST running request and retry.
                victim = self.running[-1]
                if victim is req:
                    break  # nothing left to kill; stop scheduling
                self.running.pop()
                self._preempt(victim, to_fit=req.req_id)
                continue  # retry this request, do not advance

            items.append(self._make_item(req, n))
            budget -= n
            i += 1

        # --- b. admit new requests, never at a running one's expense ---
        while self.waiting and budget > 0 and len(self.running) < self.max_num_seqs:
            req = self.waiting[0]

            # On first sight, see if someone already computed this prefix.
            if req.computed == 0 and not req.table:
                hit = self.kv.reuse(req)
                if hit:
                    self.events.append(f"{req.req_id} prefix hit {hit}")

            n = min(req.num_tokens - req.computed, budget)
            if n <= 0:
                break

            if not self.kv.alloc(req, n):
                self.events.append(f"{req.req_id} blocked (no memory)")
                break  # <-- break, never preempt. See the module docstring.

            self.waiting.popleft()
            self.running.append(req)
            items.append(self._make_item(req, n))
            budget -= n

        return items

    def _make_item(self, req: Request, n: int) -> PlanItem:
        # alloc() already advanced `computed`, so this step's positions are
        # [computed - n, computed).
        return PlanItem(
            req=req,
            start=req.computed - n,
            num_tokens=n,
            block_ids=[b.block_id for b in req.table],
        )

    def _preempt(self, req: Request, to_fit: str) -> None:
        """Take all its memory back and put it back in line.

        This is NOT the same as finishing. The request keeps its tokens and
        will recompute them later.
        """
        self.kv.drop(req)
        self.waiting.appendleft(req)  # jump the queue: it was here first
        self.events.append(f"preempt {req.req_id} (to fit {to_fit})")

    # ------------------------------------------------------------------
    # 2. settle
    # ------------------------------------------------------------------
    def update(self, items: list[PlanItem]) -> None:
        """Apply the model's output. Sample, check stop, publish KV."""
        for item in items:
            req = item.req

            # Publish blocks that are now full, so others can reuse them.
            self.kv.publish(req)

            # The model gets logits only if this step reached the end.
            # A prefill chunk in the middle produces no token.
            if req.computed < req.num_tokens:
                continue

            req.tokens.append(self.model(req))
            req.n_output += 1

            if req.n_output >= req.max_tokens:
                self.running.remove(req)
                self.kv.drop(req)
                self.events.append(f"{req.req_id} done")

    def model(self, req: Request) -> int:
        """Stand-in for the GPU. Real one returns logits; we invent a token."""
        return 100 + len(req.tokens)

    # ------------------------------------------------------------------
    # 3. drive it
    # ------------------------------------------------------------------
    def run(self) -> None:
        step = 0
        while self.waiting or self.running:
            step += 1
            items = self.step()
            plan = " ".join(f"{i.req.req_id}:{i.num_tokens}" for i in items) or "-"
            print(
                f"  step {step:2d} | {plan:<14} | running="
                f"{[r.req_id for r in self.running]}"
                f" free_blocks={len(self.kv.free)}"
            )
            for event in self.events:
                print(f"           |   {event}")
            self.update(items)


if __name__ == "__main__":
    # 6 blocks x 4 tokens = 24 token slots. Deliberately tight, so you can
    # watch the scheduler run out and preempt.
    sched = Scheduler(KVStore(num_blocks=6, block_size=4),
                      max_num_tokens=8, max_num_seqs=4)

    sched.add_request(Request("r0", list(range(6)), max_tokens=2))
    sched.add_request(Request("r1", list(range(20)), max_tokens=2))
    sched.add_request(Request("r2", list(range(4)), max_tokens=2))

    print("r0: 6 tokens, r1: 20 tokens, r2: 4 tokens, budget 8/step")
    sched.run()

    # When every request is done, every block must be back in the pool.
    assert not sched.waiting and not sched.running
    assert len(sched.kv.free) == 6, "block leak"
    print("  all requests done, all 6 blocks returned")
