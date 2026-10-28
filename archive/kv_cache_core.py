from collections import deque
from dataclasses import dataclass, field


def block_hash(parent, tokens):
    return hash((parent, tuple(tokens)))


class Block:
    """One page of KV memory."""

    def __init__(self, block_id):
        self.block_id = block_id  # index into the real GPU tensor
        self.ref_count = 0        # how many requests are using it
        self.hash = None          # set once its tokens are cached


class BlockPool:
    def __init__(self, num_blocks, block_size):
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]
        # Free ids. Left end is evicted first (least recently used).
        self.free = deque(range(num_blocks))
        self.hash_to_block = {}  # content hash -> block. This is the cache.

    def allocate(self):
        """Hand out one block. Any cached data it held is now lost."""
        block = self.blocks[self.free.popleft()]
        if block.hash is not None:
            del self.hash_to_block[block.hash]
            block.hash = None
        block.ref_count = 1
        return block

    def release(self, block):
        """Drop one user. At zero the block is free, but keeps its hash so
        its content can still be reused."""
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free.append(block.block_id)

    # ---------- prefix caching ----------

    def lookup(self, key):
        """Find a block that already holds this content."""
        block = self.hash_to_block.get(key)
        if block is None:
            return None
        if block.ref_count == 0:
            # It is sitting in the free list. Pull it out.
            # O(n) here. The real one uses a linked list to make it O(1).
            self.free.remove(block.block_id)
        block.ref_count += 1
        return block

    def register(self, block, key):
        """Publish a block's content so later requests can find it."""
        if block.hash is not None or key in self.hash_to_block:
            return  # already cached, or same content is cached elsewhere
        block.hash = key
        self.hash_to_block[key] = block


@dataclass
class Request:
    tokens: list[int]
    table: list[Block] = field(default_factory=list)
    # How many tokens' KV we actually have. The gap
    #     len(tokens) - computed
    # is the work left to do. Big gap = prefill, gap of 1 = decode.
    computed: int = 0


class KVCacheManager:
    """Per-request block tables on top of a BlockPool."""

    def __init__(self, num_blocks, block_size):
        self.block_size = block_size
        self.pool = BlockPool(num_blocks, block_size)
        self.requests = {}

    def add_request(self, req_id, tokens):
        self.requests[req_id] = Request(list(tokens))

    def free(self, req_id):
        """End the request. Blocks go back to the pool but stay cached."""
        for block in self.requests.pop(req_id).table:
            self.pool.release(block)

    def allocate_slots(self, req_id, n):
        """Reserve room for n more tokens.

        True  -> blocks reserved, `computed` advanced.
        False -> nothing changed. (So the scheduler can preempt and retry.)
        """
        req = self.requests[req_id]
        need = -(-(req.computed + n) // self.block_size)  # ceil division
        extra = need - len(req.table)

        if extra <= 0:
            req.computed += n
            return True
        if extra > len(self.pool.free):
            return False  # check before changing anything: failure is atomic

        for _ in range(extra):
            req.table.append(self.pool.allocate())
        req.computed += n
        return True

    def block_table(self, req_id):
        """The ids the GPU needs. This becomes an int32 tensor each step."""
        return [b.block_id for b in self.requests[req_id].table]

    # ---------- prefix caching ----------

    def reuse_prefix(self, req_id):
        """Adopt blocks for the longest cached prefix. Returns tokens reused.

        Call once, right after add_request().
        """
        req = self.requests[req_id]
        parent = None
        reused = 0
        # NOTE the `- 1`: never take the block holding the LAST token.
        # We always need one token left to run the model on, or there is
        # no forward pass and no logits to sample from.
        while reused + self.block_size <= len(req.tokens) - 1:
            key = block_hash(parent, req.tokens[reused : reused + self.block_size])
            block = self.pool.lookup(key)
            if block is None:
                break
            req.table.append(block)
            parent = key
            reused += self.block_size
        req.computed = reused
        return reused

    def cache_blocks(self, req_id):
        """After the model runs, publish every block that is now full.

        Only full blocks can be cached. A half-full block's content is not
        a prefix anyone else can use.
        """
        req = self.requests[req_id]
        parent = None
        for i, block in enumerate(req.table):
            end = (i + 1) * self.block_size
            if end > req.computed:
                break
            key = block_hash(parent, req.tokens[end - self.block_size : end])
            self.pool.register(block, key)
            parent = key  # keep the chain going even on a cache hit


if __name__ == "__main__":
    m = KVCacheManager(num_blocks=8, block_size=4)

    m.add_request("a", list(range(10)))  # 10 tokens -> 3 blocks
    assert m.reuse_prefix("a") == 0
    assert m.allocate_slots("a", 10)
    m.cache_blocks("a")
    assert m.block_table("a") == [0, 1, 2]

    m.add_request("b", list(range(10)))  # same prompt
    assert m.reuse_prefix("b") == 8      # 2 blocks shared, 1 token left to run
    assert m.requests["b"].table[0].ref_count == 2

    m2 = KVCacheManager(num_blocks=1, block_size=4)
    m2.add_request("x", list(range(8)))
    assert m2.allocate_slots("x", 8) is False
    assert m2.block_table("x") == []     # failed cleanly, nothing allocated

    print("ok")
