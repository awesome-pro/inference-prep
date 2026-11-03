import torch
from dataclasses import dataclass, field
from collections import deque


class Block:
    __slots__ = ("block_id", "hash", "ref_count")

    def __init__(self, block_id: int) -> None:
        self.block_id = block_id
        self.hash: int | None = None
        self.ref_count = 0



class KVCacheManager:
    def __init__(self, num_blocks: int, block_size: int) -> None:
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.free_blocks: deque[int] = deque(range(num_blocks))
        self.cached: dict[int, Block] = {}
        self.req_to_blocks: dict[str,list[int]] = {}

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def _take(self) -> int:
        block = self.blocks[self.free_blocks.popleft()]
        if block.hash is not None:
            del self.cached[block.hash]
            block.hash = None
        block.ref_count += 1
        return block.block_id

    def _release(self, block_id: int) -> None:
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_blocks.append(block_id)

    def get_computed_blocks(self, request: Request) -> tuple[list[int], int]:
        block_ids = []
        n = 0
        parent = None

        while n + self.block_size <= len(request.tokens) - 1:
            key = hash((parent, tuple(request.tokens[n: n + self.block_size])))
            block = self.cached.get(key)
            if block is None:
                break
            if block.ref_count == 0:
                self.free_blocks.remove(block.block_id)

            block.ref_count += 1
            block_ids.append(block.block_id)
            parent = key
            n += self.block_size

        return block_ids, n


    def allocate_slots(self, request: Request, num_new_tokens: int, adopted_block_ids: list[int] = []) -> list[int] | None:
        num_computed = request.num_computed_tokens + self.block_size * len(adopted_block_ids)
        required_blocks = -(-(num_computed + num_new_tokens) // self.block_size)

        existing_blocks = len(self.req_to_blocks.get(request.req_id, []))
        num_new_blocks = required_blocks - existing_blocks - len(adopted_block_ids)

        if num_new_blocks > self.num_free_blocks:
            for block_id in adopted_block_ids:
                self._release(block_id)
            return None

        allocated = list(adopted_block_ids)
        for _ in range(num_new_blocks):
            allocated.append(self._take())
        self.req_to_blocks.setdefault(request.req_id, []).extend(allocated)
        self._cache_full_blocks(request, up_to=num_computed)
        return allocated
        

    def _cache_full_blocks(self, request: Request, up_to: int) -> None:
        parent = None
        for i, block_id in enumerate(self.req_to_blocks[request.req_id]):
            end = (i + 1) * self.block_size
            if up_to and end > up_to:
                break
            key = hash((parent, tuple(request.tokens[end - self.block_size: end])))
            block = self.blocks[block_id]

            if block.hash is None and key not in self.cached:
                block.hash = key
                self.cached[key] = block
            parent = key


    def free(self, request: Request) -> None:
        for block_id in self.req_to_blocks.get(request.req_id, []):
            self._release(block_id)

    def block_table(self, req_id: str) -> list[int]:
        return list(self.req_to_blocks.get(req_id, []))


WAITING, RUNNING, PREEMPTED, DONE = "waiting", "running", "preempted", "done"


@dataclass
class Request:
    req_id: str
    tokens: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0
    status: str = WAITING
    max_tokens: int = 8
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

class Scheduler:
    def __init__(self, kv: KVCacheManager, max_num_tokens: int, max_num_seqs: int) -> None:
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

    def _prempt(self, req: Request) -> None:
        self.kv.free(req)
        req.status = PREEMPTED
        req.num_computed_tokens = 0
        self.running.remove(req)
        self.waiting.appendleft(req)


    def schedule(self)-> SchedulerOutput:
        out = SchedulerOutput()
        self.events = []
        budget = self.max_num_tokens

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

            new_blocks = self.kv.allocate_slots(request=req, num_new_tokens=n)
            if new_blocks is None:
                # preempt the latest request in running to the get more space for this request
                victim = self.running[-1]
                if victim is req:
                    break
                self.running.pop()
                self._prempt(victim)
                out.preempted_req_ids.add(victim.req_id)
                continue

            deltas[req.req_id] = new_blocks
            num_scheduled[req.req_id] = n
            scheduled_running.append(req)

            budget -= n
            i += 1

        # execute the waiting reqs
        while self.waiting and budget > 0 and len(self.running) < self.max_num_seqs:
            req = self.waiting[0]
            if req.num_computed_tokens == 0:
                adopted_block_ids, num_computed = self.kv.get_computed_blocks(req)
            else:
                adopted_block_ids, num_computed = [], req.num_computed_tokens

            n = min(req.num_tokens - num_computed, budget)

            new_blocks  = self.kv.allocate_slots(req, n, adopted_block_ids)
            if new_blocks is None:
                self.events.append(f"Memory out. {req.req_id} blocked")
                break

            # new blocks allocated
            self.waiting.pop()
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
             token_ids=list(req.tokens),
             block_ids=self.kv.block_table(req.req_id),
             num_computed_tokens=req.num_computed_tokens
            ) for req in scheduled_new
        ]
        out.num_scheduled_tokens = num_scheduled
        out.block_deltas = deltas
        out.finished_req_ids = self.finished_req_ids

        # self._update_after_schedule()
        for req_id, n in out.num_scheduled_tokens.items():
            self.requests[req_id].num_computed_tokens += n
        self.finished_req_ids = set()

        return out

    def _free_request(self, req: Request) -> None:
        if req in self.running:
            self.running.remove(req)
        self.kv.free(req)
        req.status = DONE
        self.requests.pop(req.req_id)
        self.finished_req_ids.add(req.req_id)
        self.events.append(f"{req.req_id} done")

    def update_from_output(self, output: SchedulerOutput, sampled: dict[str, list[int]]) -> None:
        for req_id in output.num_scheduled_tokens:
            req = self.requests[req_id]
            if req.num_computed_tokens < req.num_tokens:
                continue
            for token in sampled.get(req_id, ()):
                req.tokens.append(token)
                req.n_output += 1

            if req.n_output >= req.max_tokens:
                self._free_request(req)
            