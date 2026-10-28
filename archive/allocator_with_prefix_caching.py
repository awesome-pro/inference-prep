from __future__ import annotations

import hashlib
import struct
from collections import OrderedDict, deque
from dataclasses import dataclass, field


@dataclass
class RequestState:
    tokens: tuple[int, ...]
    scope: str
    table: list[int] = field(default_factory=list)
    reserved_tokens: int = 0
    computed_tokens: int = 0


class KVAllocator:
    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        
        self._empty = deque(range(num_blocks))
        self._refcounts = [0] * num_blocks
        self._requests: dict[str, RequestState] = {}
        # digest -> physical block; indexed blocks are immutable.
        self._cache: dict[bytes, int] = {}
        self._key_of: dict[int, bytes] = {}
        # Only zero-refcount cached blocks are evictable, oldest first.
        self._evictable: OrderedDict[bytes, None] = OrderedDict()

    def create(self, request_id: str, tokens: list[int], *, scope: str = "model-v1") -> None:
        if request_id in self._requests:
            raise ValueError(f"duplicate request {request_id}")
        if not tokens or any(not isinstance(t, int) or t < 0 or t >= 2**64 for t in tokens):
            raise ValueError("tokens must be a non-empty list of unsigned 64-bit ids")
        self._requests[request_id] = RequestState(tuple(tokens), scope)

    def _hash(self, scope: str, previous: bytes, block_tokens: tuple[int, ...]) -> bytes:
        # The chained hash binds a block to every preceding block and the model scope.
        h = hashlib.blake2b(digest_size=16)
        name = scope.encode("utf-8")
        h.update(len(name).to_bytes(4, "little"))
        h.update(name)
        h.update(previous)
        for token_id in block_tokens:
            h.update(struct.pack("<Q", token_id))
        return h.digest()

    def attach_prefix(self, request_id: str) -> int:
        """Reuse longest cached full-block prefix, before any private allocation.

        Returns number of *tokens* whose KV can be reused.
        """
        req = self._requests[request_id]
        if req.table or req.reserved_tokens or req.computed_tokens:
            raise ValueError("attach_prefix must run on a fresh request")
        previous = b""
        block_ids: list[int] = []
        for pos in range(0, len(req.tokens) - self.block_size + 1, self.block_size):
            token_chunk = req.tokens[pos:pos + self.block_size]
            digest = self._hash(req.scope, previous, token_chunk)
            block = self._cache.get(digest)
            if block is None:
                break
            block_ids.append(block)
            previous = digest

        for block in block_ids:
            # Previously unowned cached KV becomes pinned for this request.
            if self._refcounts[block] == 0:
                del self._evictable[self._key_of[block]]
            self._refcounts[block] += 1
            req.table.append(block)
        reused = len(block_ids) * self.block_size
        req.reserved_tokens = reused
        req.computed_tokens = reused
        return reused

    def available_blocks(self) -> int:
        """Empty blocks + zero-owner cached blocks that may be evicted."""
        return len(self._empty) + len(self._evictable)

    def _take_block(self) -> int:
        if self._empty:
            return self._empty.popleft()
        # Recycle the oldest *unowned* cached block. Held blocks never evict.
        key, _ = self._evictable.popitem(last=False)
        block = self._cache.pop(key)
        del self._key_of[block]
        assert self._refcounts[block] == 0
        return block

    def allocate(self, request_id: str, target_tokens: int) -> list[int]:
        """Grow reserved capacity to `target_tokens`; does not generate KV."""
        req = self._requests[request_id]
        if target_tokens < req.reserved_tokens or target_tokens > len(req.tokens):
            raise ValueError("target must grow monotonically and fit the request")
        required = (target_tokens + self.block_size - 1) // self.block_size
        extra = required - len(req.table)
        if extra > self.available_blocks():
            raise MemoryError("insufficient KV blocks: nothing allocated")
        for _ in range(extra):
            block = self._take_block()
            self._refcounts[block] = 1
            req.table.append(block)
        req.reserved_tokens = target_tokens
        return req.table.copy()

    def mark_computed(self, request_id: str, target_tokens: int) -> None:
        """Publish only completed KV blocks after the runner has written them."""
        req = self._requests[request_id]
        if not req.computed_tokens <= target_tokens <= req.reserved_tokens:
            raise ValueError("computed cursor must increase within reserved capacity")
        req.computed_tokens = target_tokens
        previous = b""
        full = target_tokens // self.block_size
        for logical in range(full):
            start = logical * self.block_size
            digest = self._hash(req.scope, previous, req.tokens[start:start + self.block_size])
            previous = digest
            block = req.table[logical]
            old = self._key_of.get(block)
            if old is not None:
                if old != digest:
                    raise AssertionError("cached immutable block has mismatching tokens")
                continue
            # Another physical block may already cache the identical content.
            if digest not in self._cache:
                self._cache[digest] = block
                self._key_of[block] = digest

    def free(self, request_id: str) -> None:
        """Drop this request's references; keep full cached blocks for reuse."""
        req = self._requests.pop(request_id)  # repeated free -> KeyError
        for block in req.table:
            assert self._refcounts[block] > 0
            self._refcounts[block] -= 1
            if self._refcounts[block] == 0:
                key = self._key_of.get(block)
                if key is not None:
                    self._evictable[key] = None
                else:
                    self._empty.append(block)

    def block_table(self, request_id: str) -> list[int]:
        return self._requests[request_id].table.copy()

    def refcount(self, block: int) -> int:
        return self._refcounts[block]

    def assert_invariants(self) -> None: