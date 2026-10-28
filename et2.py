from collections import deque
from dataclasses import dataclass, field

import torch

HEAD_DIM = 8
VOCAB = 8

class Block:
    __slots__ = ("block_id", "ref_count", "hash")

    def __init__(self, block_id: int) -> None:
        self.block_id = block_id
        self.ref_count = 0
        self.hash: int | None = None

class KVCacheManager:
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

    def _release(self, block_id):
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_blocks.append(block_id)

    def get_computed_blocks(self, request: Request) -> tuple[list[int], int]:
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

    def allocate_slots(self, request: Request, num_new_tokens: int, adopted: list[int] = []) -> list[int] | None:
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

    def _cache_full_blocks(self, request: Request, up_to: int) -> None:
        parent = None
        for i, block_id in enumerate(self.req_to_blocks[request.req_id]):
            end = (i + 1) * self.block_size
            if end > up_to:
                break
            key = hash((parent, tuple(request.tokens[ end - self.block_size: end])))
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
    

# request + plan
WAITING, RUNNING, PREEMPTED, DONE = "waiting", "running", "preempted", "done"


@dataclass
class Request:
    req_id: str
    tokens: list[int]
    max_tokens: int = 8
    num_computed_tokens: int = 0
    status: str = WAITING
    n_output = 0

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

# scheduler

class Schduler:
    def __init__(self, kv: KVCacheManager, max_num_tokens: int, max_num_seqs: int) -> None:
        self.kv = kv

        self.max_num_tokens = max_num_tokens
        self.max_num_seqs = max_num_seqs

        self.requests: dict[str, Request] = {}

        self.waiting: deque[Request] = deque()
        self.running: list[Request] =[]
        self.finished_req_ids: set[str] = set()

        self.events: list[str] = []

    def add_request(self, req: Request) -> None:
        self.requests[req.req_id] = req
        self.waiting.append(req)

    def schedule(self) -> SchedulerOutput:
        self.events = []
        budget = self.max_num_tokens

        out = SchedulerOutput()

        scheduled_running: list[Request] = []
        scheduled_new: list[Request] = []
        deltas: dict[str, list[int]] = {}
        num_scheduled: dict[str, int] = {}

        i=0
        while i < len(self.running) and budget > 0:
            req = self.running[i]
            n = min(req.num_tokens - req.num_computed_tokens, budget)

            if n <= 0:
                i += 1
                continue

            new_block_ids = self.kv.allocate_slots(request=req, num_new_tokens=n)
            if new_block_ids is None:
                # Preempt the victim
                victim = self.running[-1]
                if victim is req:
                    break

                self.running.pop()
                self._preempt(victim, out=out)
                continue
            
            deltas[req.req_id] = new_block_ids
            num_scheduled[req.req_id] = n
            scheduled_running.append(req)
            budget -= n
            i += 1

        # Go to the waiting queue

        while self.waiting  and budget > 0 and len(self.running) < self.max_num_seqs:
            req = self.waiting[0]

            # If it's a completely new request
            if req.num_computed_tokens == 0:
                adopted, num_computed = self.kv.get_computed_blocks(request=req)
                if num_computed:
                    self.events.append(f"Prefix hit {req.req_id} {num_computed}")
            else:
                adopted, num_computed = [], req.num_computed_tokens

            n  = min(req.num_tokens - req.num_computed_tokens, budget)
            new_block_ids = self.kv.allocate_slots(request=req, num_new_tokens=n, adopted=adopted)
            if new_block_ids is None:
                # memoy out
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
             req_id=req.req_id,
             token_ids=req.tokens,
             block_ids=self.kv.block_table(request=req),
             num_computed_tokens=req.num_computed_tokens
            ) for req in scheduled_new
        ]

        out.block_deltas = deltas
        out.num_scheduled_tokens = num_scheduled
        out.finished_req_ids = self.finished_req_ids

        for req_id, n in out.num_scheduled_tokens.items():
            self.requests[req_id].num_computed_tokens += n
        self.finished_req_ids = set()

        return out

    def _preempt(self, req: Request, out: SchedulerOutput) -> None:
        self.kv.free(request=req)
        req.num_computed_tokens = 0
        req.status = WAITING
        self.waiting.appendleft(req)
        out.preempted_req_ids.add(req.req_id)
        self.events.append(f"preempt {req.req_id}")

    def update_from_output(self, out: SchedulerOutput, sampled: dict[str, list[int]]) -> None:
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

        self.kv.free(request=req)
        req.status = DONE
        self.requests.pop(req.req_id, None)
        self.finished_req_ids.add(req.req_id)
        self.events.append(f"{req.req_id} done")

# gpu model runner

@dataclass
class CachedRequestState:
    req_id: str
    token_ids: list[int]
    block_ids: list[int]
    num_computed_tokens: int


class GPUModelRunner:
    def __init__(self, num_blocks: int, block_size: int, max_num_tokens: int) -> None:
        self.block_size = block_size

        self.kv_cache = torch.zeros(num_blocks, block_size, 2, HEAD_DIM)

        self.input_ids = torch.zeros(max_num_tokens, dtype=torch.float32)
        self.positions = torch.zeros(max_num_tokens, dtype=torch.int64)
        self.slot_mapping = torch.zeros(max_num_tokens, dtype=torch.int64)

        g = torch.Generator().manual_seed(0)

        self.embed = torch.randn(VOCAB, HEAD_DIM, generator=g)
        self.wq = torch.randn(HEAD_DIM, HEAD_DIM, generator=g) * 0.3
        self.wk = torch.randn(HEAD_DIM, HEAD_DIM, generator=g) * 0.3
        self.wv = torch.randn(HEAD_DIM, HEAD_DIM, generator=g) * 0.3
        self.lm_head = torch.randn(HEAD_DIM, VOCAB, generator=g)

        self.verify = True
        self.states: dict[str, CachedRequestState] = {}

    def execute(self, out: SchedulerOutput) -> dict[str, list[int]]:
        pass

    def _update_states(self, out: SchedulerOutput) -> None:
        for req_id in out.finished_req_ids | out.preempted_req_ids:
            self.states.pop(req_id, None)

        for data in out.new_reqs:
            self.states[data.req_id] = CachedRequestState(
                req_id=data.req_id,
                token_ids=list(data.token_ids),
                block_ids=list(data.block_ids),
                num_computed_tokens=data.num_computed_tokens
            )

        for req_id, new_block_ids in out.block_deltas.items():
            self.states[req_id].block_ids.extend(new_block_ids)