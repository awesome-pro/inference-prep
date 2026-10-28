from collections import deque
from dataclasses import dataclass, field

class Block:
    def __init__(self, block_id: int) -> None:
        self.block_id = block_id
        self.hash = None
        self.ref_count += 1


class BlockPool:
    def __init__(self, block_size: int, num_blocks: int) -> None:
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]

        self.free = deque(range(num_blocks))

        self.hash_to_block: dict[int, Block] = {}


    def allocate(self) -> Block:
        pass

    def release(self, req_id: int) -> None:
        pass

    def lookup(self, key: int) -> Block | None:
        pass

    def register(self, block: Block, key: int) -> bool:
        pass

@dataclass
class Request:
    req_id: int
    tokens: list[int]
    block_table: list[Block] = field(default_factory=list)
    num_computed_tokens: int = 0

    @property
    def num_tokens(self) -> int:
        return len(self.tokens)

class KVCacheManager:
    def __init__(self, num_blocks: int, block_size: int) -> None:
        self.block_size = block_size

        self.pool = BlockPool(num_blocks=num_blocks, block_size=block_size)

        self.requests: dict[int, Request] = {}


    def add_request(self, req_id: int, tokens: list[int]) -> Request:
        pass

    def reuse_prefix(self, req_id) -> int:
        pass

    def allocate_slots(self, req_id, num_new_tokens) -> bool:
        pass

    def cache_all_blocks(self, req_id) -> int:
        pass

    def free(self, req_id) -> None:
        pass

@dataclass
class PlanItem:
    req: Request
    start: int
    num_tokens: int
    block_ids: list[int]

class Scheduler:
    def __init__(self, kv: KVCacheManager, max_num_tokens: int, max_num_seqs: int) -> None:
        self.kv = kv
        self.max_num_tokens = max_num_tokens
        self.max_num_seqs = max_num_seqs

        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []

        self.events: list[str] = []

    def _make_item(self, req: Request, n: int) -> PlanItem:
            # alloc() already advanced `computed`, so this step's positions are
            # [computed - n, computed).
            return PlanItem(
                req=req,
                start=req.num_computed_tokens - n,
                num_tokens=n,
                block_ids=[b.block_id for b in req.block_table],
            )

    def add_request(self, req: Request) -> None:
        self.waiting.append(req)
        return

    def step(self) -> list[PlanItem]:
        self.events = []
        budget = self.max_num_tokens
        items: list[PlanItem] = []

        # allocating tokens to running requests
        i = 0
        while i < len(self.running) and budget > 0:
            req = self.running[i]
            n = min((req.num_tokens - req.num_computed_tokens), budget)
            if n <= 0:
                i += 1
                continue
            # if tokens can't be allocated, preemt the newest request
            if not self.kv.allocate_slots(req_id=req.req_id, num_new_tokens=n):
                victim = self.running[-1]
                if victim is req:
                    break # nothin left to kill, stop scheduling

                self.running.pop()
                self._preempt(victim, to_fit=req.req_id)
                continue

            items.append(self._make_item(req, n))
            budget -= n
            i += 1


        while self.waiting and budget > 0 and len(self.running) < self.max_num_seqs:
            req = self.waiting[0]

            if req.num_computed_tokens == 0 and not req.block_table:
                hit = self.kv.reuse_prefix(req_id=req.req_id)
                if hit:
                    self.events.append(f"{req.req_id} prefix hit")

                n = min(req.num_tokens - req.num_computed_tokens, budget)
                if n <= 0:
                    break

                if not self.kv.allocate_slots(req.req_id, n):
                    self.events.append(f"{req.req_id} blocked (no memory)")
                    break
                self.waiting.popleft()
                self.running.append(req)
                items.append(self._make_item(req, n))
                budget -= n

        return items

    def _preemt(self, req: Request, to_fit: str) -> None:

        self.kv.free(req_id=req.req_id)
        self.waiting.appendleft(req)
        self.events.append(f"preempt {req.req_id} to (to fit {to_fit})")