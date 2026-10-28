from collections import deque
from dataclasses import field, dataclass


def hash_block(parent: int | None, token_ids: tuple[int, ...]) -> int:
    return hash((parent, token_ids)) 

class Block:
    def __init__(self, block_id: int) -> None:
        self.block_id = block_id
        self.ref_count = 0
        self.hash: int | None = None

class BlockPool:
    def __init__(self, num_blocks: int, block_size: int) -> None:
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]

        # queue of the free block ids
        self.free_block_queue = deque(range(num_blocks))

        # req id -> hash
        self.hash_to_block: dict[int, Block] = {}

    def allocate(self) -> Block:
        block_id = self.free_block_queue.popleft()
        block = self.blocks[block_id]
        if block.hash is not None:
            del self.hash_to_block[block_id]
            block.hash = None
        return block

    def lookup(self, key) -> Block | None:
        if key not in self.hash_to_block:
            return None

        block = self.hash_to_block[key]
        if block.ref_count == 0:
            self.free_block_queue.remove(block.block_id)
        block.ref_count += 1
        return block

    def release(self, block: Block) -> None:
        assert block.ref_count > 0
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_block_queue.append(block.block_id)

    def register(self, block: Block, key: int) -> bool:
        if block.hash:
            return False
        if key in self.hash_to_block:
            return False
        block.hash = key
        self.hash_to_block[key] = block
        return True


@dataclass
class RequestState:
    req_id: int
    token_ids: list[int]
    block_table: list[Block] = field(default_factory=list)
    num_computed_tokens: int = 0


class KVCacheManager:
    def __init__(self, num_blocks: int, block_size: int) -> None:
        self.block_size = block_size

        # memory pool
        self.pool = BlockPool(num_blocks=num_blocks, block_size=block_size)

        # requests
        self.requests: dict[int, RequestState] = {}

        self.num_queried_tokens = 0
        self.num_hit_tokens = 0


    def add_request(self, req_id: int, token_ids: list[int]) -> RequestState:
        if req_id in self.requests:
            raise ValueError("Request already exists.")
        if not token_ids:
            raise ValueError("Request does not have any tokens.")
        request = RequestState(req_id=req_id,token_ids=token_ids)
        self.requests[req_id] = request
        return request

    def free(self, req_id) -> None:
        if req_id not in self.requests:
            raise KeyError("request not there")

        for block in self.requests[req_id].block_table:
            self.pool.release(block)

    # after admission, before allocation. returns tokens reused
    def reuse_prefix(self, req_id) -> int:
        req = self.requests[req_id]
        assert req.num_computed_tokens == 0 and not req.block_table

        parent: int | None = None
        reused = 0

        while reused + self.block_size <= len(req.token_ids) - 1:
            key = hash_block(parent=parent, token_ids=tuple(req.token_ids[reused: reused + self.block_size]))
            block = self.pool.lookup(key)
            if block is None:
                break

            req.block_table.append(block)
            parent = key
            reused += self.block_size

        req.num_computed_tokens = reused
        return reused


    def allocate_slots(self, req_id, num_new_tokens) -> bool:
        """
        True -> blocks reserved & num_computed_tokens advanced
        """
        req = self.requests[req_id]
        need = -(-(req.num_computed_tokens + num_new_tokens) // self.block_size)
        extra = need - len(req.block_table)

        if extra <= 0:
            req.num_computed_tokens += num_new_tokens
            return True
        if extra > len(self.pool.free_block_queue):
            return False

        for _ in range(extra):
            req.block_table.append(self.pool.allocate())
        req.num_computed_tokens += num_new_tokens
        return True

    # after the forward pass, publish every fully computed block
    def cache_full_blocks(self, req_id) -> int:
        """After model run, publish every block that is full."""
        req = self.requests[req_id]
        parent = None
        for i, block in enumerate(req.block_table):
            end = (i + 1) * self.block_size
            if end > req.num_computed_tokens:
                break
            key = hash_block(parent=parent, token_ids=tuple(req.token_ids[end - self.block_size : end]))
            self.pool.register(block=block,key=key)
            parent=key

    def block_table(self, req_id) -> list[int]:
        return [block.block_id for block in self.requests[req_id].block_table]

