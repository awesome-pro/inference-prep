from collections import deque
from dataclasses import dataclass, field


def block_hash(parent, tokens):
    return hash((parent, tuple(tokens)))

class Block:
    def __init__(self, block_id) -> None:
        self.block_id = block_id
        self.ref_count = 0
        self.hash: str | None = None


class BlockPool:
    def __init__(self, num_blocks, block_size) -> None:
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]

        # Free block ids. left is evicted first
        self.free = deque(range(num_blocks))
        self.hash_to_block: dict[str, Block] = {} # content hash -> block

    def allocate(self):
        block = self.blocks[self.free.popleft()]
        if block.hash is not None:
            del self.hash_to_block[block.hash]
            block.hash = None

        block.ref_count = 1
        return block

    def release(self, block: Block):
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free.append(block.block_id)

    # ------ prefix caching ------

    def lookup(self, key):
        block = self.hash_to_block.get(key)
        if block is None:
            return None

        # if the block is in free q
        if block.ref_count == 0:
            self.free.remove(block.block_id)

        block.ref_count += 1
        return block

    def register(self, block: Block, key: str):
        if block.hash is not None or key in self.hash_to_block:
            return
        block.hash = key
        self.hash_to_block[key] = block


@dataclass
class Request:
    tokens: list[int]
    table: list[Block] = field(default_factory=list)

    num_computed_tokens: int = 0


class KVCacheManger:
    def __init__(self, num_blocks, block_size) -> None:
        self.block_size = block_size
        self.pool = BlockPool(num_blocks=num_blocks, block_size=block_size)
        self.requests = {}

    def add_request(self, req_id, tokens):
        self.requests[req_id] = Request(list(tokens))

    def free(self, req_id):
        for block in self.requests.pop(req_id).table:
            self.pool.release(block=block)

    def allocate_slots(self, req_id, n):
        req = self.requests[req_id]
        need = -(-(req.computed + n) // self.block_size)
        extra = need - len(req.table)

        if extra <= 0:
            req.computed += n
            return True
        if extra > len(self.pool.free):
            return False

        for _ in range(extra):
            req.table.append(self.pool.allocate())

        req.computed += n
        return True

    def reuse_prefix(self, req_id):
        req = self.requests[req_id]
        parent = None
        reused = 0

        while reused + self.block_size <= len(req.tokens) - 1:
            key = block_hash(parent, req.tokens[reused : reused + self.block_size])
            block = self.pool.lookup(key)
            if block is None:
                break
