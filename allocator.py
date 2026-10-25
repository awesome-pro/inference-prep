from collections import deque, defaultdict

class KVBlock:
    def __init__(self, id: int, block_size = 4) -> None:
        self.id = id
        self.block_size: int = block_size
        self.ref_count: int = 0

class KVBlockAllocator:
    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError()

        self.num_blocks = num_blocks
        self.block_size = block_size

        # Available physical block ids
        self._free = deque([KVBlock])

        # map of request id -> ordered physical ids
        self._tables = defaultdict(list[int])

        # request id -> number of token positions reserved
        self._reserved: dict[str, int] = {}


    def allocate(self, request_id, num_tokens):
        if self.num_blocks <= 0:
            raise ValueError

        previous_tokens = self._reserved.get(request_id, 0)
        if num_tokens < previous_tokens:
            raise ValueError("allocate cannot shrink a requst")

        # number of the blocks reqd for the target token count

        reqd = (
            num_tokens + self.block_size - 1
        ) // self.block_size

        current_table = self._tables.get(request_id, [])
        extra = reqd - len(current_table)

        if extra > len(self._free):
            raise MemoryError("Not enough KV blocks")

        # Commit allocation
        table = self._tables.setdefault(request_id, [])

        for _ in range(extra):
            block_id = self._free.popleft()
            table.append(block_id)

        self._reserved[request_id] = num_tokens
        return table.copy()

    def free(self, request_id) -> None:
        if request_id not in self._tables:
            raise KeyError("request not in memory")

        blocks = self._tables.get(request_id, [])

        # return the blocks to the free memory
        self._free.extend(blocks)
        del self._tables[request_id]

    def get_block_table(self, request_id):
        return self._tables[request_id].copy()

    def num_free_blocks(self):
        return len(self._free)


allocator = KVBlockAllocator(
    num_blocks=10,
    block_size=16,
)

assert len(allocator.allocate("r0", 33)) == 3
assert allocator.num_free_blocks() == 7

assert len(allocator.allocate("r1", 40)) == 3
assert allocator.num_free_blocks() == 4

# Still fits in existing 3 blocks
assert len(allocator.allocate("r0", 48)) == 3
assert allocator.num_free_blocks() == 4

# Crosses a block boundary
assert len(allocator.allocate("r0", 49)) == 4
assert allocator.num_free_blocks() == 3

# Release r1's three blocks
allocator.free("r1")
assert allocator.num_free_blocks() == 6

# Failure must not partially mutate state
before = allocator.num_free_blocks()

try:
    allocator.allocate("too_large", 1000)
except MemoryError:
    pass

assert allocator.num_free_blocks() == before
assert "too_large" not in allocator._tables

print("All tests passed!")

