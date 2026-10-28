#!/usr/bin/env python3
"""
kv_cache_manager.py -- a paged KV-cache allocator, sized for a 60-minute interview.
==================================================================================

THE PROBLEM
-----------
Serving an LLM means storing the K and V tensors of every token already
processed. Without that cache, each decode step would re-run the whole prompt.
The naive allocation is

    max_model_len * bytes_per_token   per request

which for a 70B model is ~10 GB per request. Dead on arrival.

The fix is **paged attention**: cut the cache into fixed-size blocks
(16 tokens each), give each request a *block table* (logical block -> physical
block), and let the attention kernel gather K/V through that table -- exactly
like an OS page table. Memory is then allocated in 16-token granules, on
demand, and two requests with the same prefix can literally *share* blocks.

THE CAST (vLLM name -> this file)
---------------------------------
    KVCacheBlock          -> KVCacheBlock    one page
    FreeKVCacheBlockQueue -> FreeBlockQueue  O(1) LRU list of idle pages
    BlockPool             -> BlockPool       owns every page + prefix hash map
    KVCacheManager        -> KVCacheManager  per-request tables + policy
    Scheduler             -> not here        decides WHO allocates and who to
                                             preempt when we return False

Allocator vs manager is the distinction worth stating out loud in the room:
the pool is pure memory and knows nothing about requests; the manager is
policy and knows nothing about eviction. The scheduler knows about both.

THE ONE INVARIANT THAT MAKES IT ALL WORK
----------------------------------------
    block.ref_count == 0  <=>  block is in the free queue

A block with ref_count 0 but a live hash is *simultaneously free and cached*:
available for someone else, but reusable for free if the same prefix shows up
again. Every function below either preserves that invariant or is a bug.
Keeping the free queue a doubly linked list is what lets a block leave it
from the *middle* in O(1) -- which is exactly what a prefix-cache hit does.

WHAT THE INTERVIEWER IS ACTUALLY TESTING
----------------------------------------
    1. Why blocks instead of a flat buffer per request?
       -> fragmentation, and the possibility of sharing.
    2. Can you keep ref_count and the free list consistent?
       -> the invariant above.
    3. Do you know eviction (allocator, invisible to callers) from
       preemption (scheduler, very visible to one request)?
       -> most candidates conflate these.
    4. Does allocate_slots() fail *atomically*?
       -> check availability first, then commit. Never half-allocate.
    5. Do you know why the last block is never cached?
       -> you need one token left to forward.

Run this file directly to see four scenarios execute.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def ceil_div(a: int, b: int) -> int:
    """How many blocks of size `b` are needed to hold `a` tokens."""
    return (a + b - 1) // b


def block_hash(parent: int | None, token_ids: tuple[int, ...]) -> int:
    """Key for a block, *chained* to the key of the block before it.

    Because the key folds in `parent`, a block's key covers the entire prefix
    leading up to it. Two requests sharing a system prompt therefore produce
    identical keys for those blocks and only diverge at the first block where
    their tokens differ.

    Interview note: `hash()` is fine for one process. vLLM uses blake2b
    because keys must agree across processes and machines (the prefix cache is
    shared with KV connectors / other nodes). Say that out loud.
    """
    return hash((parent, token_ids))


# ---------------------------------------------------------------------------
# 1. the block
# ---------------------------------------------------------------------------
class KVCacheBlock:
    """One page of KV memory. `block_id` indexes into the GPU-side tensor."""

    # __slots__ keeps millions-of-objects memory sane; free in an interview.
    __slots__ = ("block_id", "ref_count", "block_hash", "prev_free", "next_free")

    def __init__(self, block_id: int) -> None:
        self.block_id = block_id
        self.ref_count = 0
        # Set once the block's tokens have been computed and published to the
        # prefix cache. None means "holds nobody's content".
        self.block_hash: int | None = None
        self.prev_free: KVCacheBlock | None = None
        self.next_free: KVCacheBlock | None = None

    def __repr__(self) -> str:
        return f"<blk {self.block_id} rc={self.ref_count}>"


# ---------------------------------------------------------------------------
# 2. the free list (LRU, O(1) remove from anywhere)
# ---------------------------------------------------------------------------
class FreeBlockQueue:
    """Doubly linked list of idle blocks, least-recently-freed at the head.

    Why a linked list and not a deque? A block can leave the free list from the
    middle: a prefix hit wakes up a block that was freed long ago. `deque` has
    no O(1) arbitrary removal. This is the reason vLLM has its own queue class
    instead of just using `collections.deque`.

    Head = oldest = first to be evicted (content least likely to be reused).
    Tail = newest = the block we just freed.
    """

    def __init__(self) -> None:
        # Sentinels remove every `if node is None` branch from the code.
        self._head = KVCacheBlock(-1)
        self._tail = KVCacheBlock(-1)
        self._head.next_free = self._tail
        self._tail.prev_free = self._head
        self._size = 0

    def append(self, block: KVCacheBlock) -> None:
        """Push a freshly freed block onto the tail."""
        assert block.prev_free is None and block.next_free is None, "already queued"
        last = self._tail.prev_free
        assert last is not None
        last.next_free = block
        block.prev_free = last
        block.next_free = self._tail
        self._tail.prev_free = block
        self._size += 1

    def popleft(self) -> KVCacheBlock:
        """Pop the eviction victim."""
        if self._size == 0:
            raise IndexError("no free blocks")
        block = self._head.next_free
        assert block is not None
        self.remove(block)
        return block

    def remove(self, block: KVCacheBlock) -> None:
        """O(1) removal from the middle -- used by prefix-cache wake-ups."""
        prev, nxt = block.prev_free, block.next_free
        if prev is None or nxt is None:
            raise ValueError("block is not in the free queue")
        prev.next_free = nxt
        nxt.prev_free = prev
        block.prev_free = block.next_free = None
        self._size -= 1

    def __len__(self) -> int:
        return self._size

    def __bool__(self) -> bool:
        return self._size > 0


# ---------------------------------------------------------------------------
# 3. the allocator (pure memory; knows nothing about requests)
# ---------------------------------------------------------------------------
class BlockPool:
    """Owns every KV page: free list, ref counts, prefix hash map."""

    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be positive")
        self.block_size = block_size
        self.blocks = [KVCacheBlock(i) for i in range(num_blocks)]
        self.free_queue = FreeBlockQueue()
        for block in self.blocks:
            self.free_queue.append(block)
        # chained block hash -> the one physical block holding that content
        self.hash_to_block: dict[int, KVCacheBlock] = {}
        self.num_evictions = 0

    @property
    def num_free_blocks(self) -> int:
        """Capacity check. Note this counts cached-but-idle blocks too --
        they are exactly as evictable as never-used ones."""
        return len(self.free_queue)

    # -- allocation ---------------------------------------------------------
    def _take(self) -> KVCacheBlock:
        """Hand out one block, destroying whatever content it cached.

        THIS is eviction, and it is invisible to callers: nobody asked us to
        free anything, we just reused the least valuable page.
        """
        block = self.free_queue.popleft()
        if block.block_hash is not None:
            self.hash_to_block.pop(block.block_hash, None)
            block.block_hash = None
            self.num_evictions += 1
        block.ref_count = 1
        return block

    def allocate(self) -> KVCacheBlock:
        """Allocate one block. Caller must have checked `num_free_blocks`."""
        return self._take()

    def lookup(self, key: int) -> KVCacheBlock | None:
        """Prefix-cache hit: claim an idle block whose content matches `key`."""
        block = self.hash_to_block.get(key)
        if block is None:
            return None
        if block.ref_count == 0:
            # It is sitting in the free queue; wake it up. O(1) thanks to the
            # doubly linked list. This is the whole point of that data structure.
            self.free_queue.remove(block)
        block.ref_count += 1
        return block

    # -- release / publish --------------------------------------------------
    def release(self, block: KVCacheBlock) -> None:
        """Drop one reference. At zero the block becomes idle but keeps its
        hash, so it stays reusable until someone evicts it."""
        assert block.ref_count > 0, f"double free on block {block.block_id}"
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_queue.append(block)

    def register(self, block: KVCacheBlock, key: int) -> bool:
        """Publish a block's content to the prefix cache."""
        if block.block_hash is not None:
            return False  # already published (it was a cache hit)
        if key in self.hash_to_block:
            return False  # identical content already cached elsewhere
        block.block_hash = key
        self.hash_to_block[key] = block
        return True


# ---------------------------------------------------------------------------
# 4. the manager (policy: per-request block tables)
# ---------------------------------------------------------------------------
@dataclass
class RequestState:
    req_id: str
    token_ids: list[int]
    block_table: list[KVCacheBlock] = field(default_factory=list)
    # How far into the sequence the KV cache is actually valid. In vLLM this
    # is the single number the scheduler trusts, and the gap
    #     num_tokens - num_computed_tokens
    # is the work it has to schedule. Prefill = big gap, decode = gap of 1,
    # chunked prefill = big gap, granted a little at a time.
    num_computed_tokens: int = 0

    def append_token(self, token_id: int) -> None:
        """The sampler produced a token; the sequence grew by one."""
        self.token_ids.append(token_id)

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)


class KVCacheManager:
    """Per-request block tables on top of a BlockPool.

    The whole API is one question:
        allocate_slots(req, n) -> True / False
    True  = "you may compute n more tokens".
    False = "no memory; scheduler, you decide who dies."
    """

    def __init__(self, num_blocks: int, block_size: int) -> None:
        self.block_size = block_size
        self.pool = BlockPool(num_blocks, block_size)
        self.requests: dict[str, RequestState] = {}
        self.num_queried_tokens = 0
        self.num_hit_tokens = 0

    # -- lifecycle ----------------------------------------------------------
    def add_request(self, req_id: str, token_ids: list[int]) -> RequestState:
        if req_id in self.requests:
            raise ValueError(f"duplicate request {req_id}")
        if not token_ids:
            raise ValueError("token_ids must be non-empty")
        req = RequestState(req_id, list(token_ids))
        self.requests[req_id] = req
        return req

    def free(self, req_id: str) -> None:
        """End a request. Blocks go back to the pool but keep their hashes --
        that is how the prefix cache survives across requests."""
        req = self.requests.pop(req_id)
        for block in req.block_table:
            self.pool.release(block)

    # -- admission ----------------------------------------------------------
    def reuse_prefix(self, req_id: str) -> int:
        """Adopt blocks for the longest already-cached prefix. Returns tokens reused.

        Call this on admission, before allocate_slots(). Cost is one hash map
        lookup per block -- O(num_tokens / block_size).
        """
        req = self.requests[req_id]
        assert req.num_computed_tokens == 0 and not req.block_table

        parent: int | None = None
        reused = 0
        # NOTE the `- 1`: never adopt a block containing the LAST token.
        # Every step needs at least one token to forward, otherwise there is
        # no forward pass and therefore no logits to sample from. This is the
        # single most common bug in hand-written versions.
        while reused + self.block_size <= req.num_tokens - 1:
            start = reused
            key = block_hash(
                parent, tuple(req.token_ids[start : start + self.block_size])
            )
            block = self.pool.lookup(key)
            if block is None:
                break
            req.block_table.append(block)
            parent = key
            reused += self.block_size

        req.num_computed_tokens = reused
        self.num_queried_tokens += req.num_tokens - 1
        self.num_hit_tokens += reused
        return reused

    # -- the one function that matters --------------------------------------
    def allocate_slots(self, req_id: str, num_new_tokens: int) -> bool:
        """Reserve blocks so `num_new_tokens` more tokens can be computed.

        Contract:
          * True  -> blocks reserved AND num_computed_tokens advanced.
          * False -> nothing was mutated. (Atomic failure; required by the
                     scheduler, which then preempts somebody and retries.)

        O(blocks added). The capacity check is what makes it atomic -- an
        interviewer will ask "what if it fails halfway?", and the answer must
        be "it can't, I check first, then commit."
        """
        req = self.requests[req_id]
        if num_new_tokens <= 0:
            raise ValueError("num_new_tokens must be positive")

        total_tokens = req.num_computed_tokens + num_new_tokens
        required = ceil_div(total_tokens, self.block_size)
        extra = required - len(req.block_table)

        if extra <= 0:
            # We are writing into a block that is already half-full.
            req.num_computed_tokens = total_tokens
            return True

        if extra > self.pool.num_free_blocks:
            return False  # <-- the ONLY backpressure signal in the system

        for _ in range(extra):
            req.block_table.append(self.pool.allocate())
        req.num_computed_tokens = total_tokens
        return True

    # -- publish ------------------------------------------------------------
    def cache_full_blocks(self, req_id: str) -> int:
        """After a forward pass, publish every *fully* computed block.

        Only full blocks are cacheable: a half-full block's hash would not
        describe reusable KV. Call this after compute, never before.
        """
        req = self.requests[req_id]
        parent: int | None = None
        published = 0
        for i, block in enumerate(req.block_table):
            end = (i + 1) * self.block_size
            if end > req.num_computed_tokens:
                break  # block not full yet
            key = block_hash(
                parent, tuple(req.token_ids[end - self.block_size : end])
            )
            if block.block_hash is None and self.pool.register(block, key):
                published += 1
            parent = key  # chain even on a hit, or later keys break
        return published

    # -- views --------------------------------------------------------------
    def block_table(self, req_id: str) -> list[int]:
        """What gets copied to the GPU as an int32 tensor each step."""
        return [b.block_id for b in self.requests[req_id].block_table]

    @property
    def num_used_blocks(self) -> int:
        return len(self.pool.blocks) - self.pool.num_free_blocks

    @property
    def num_cached_blocks(self) -> int:
        return len(self.pool.hash_to_block)

    @property
    def prefix_hit_rate(self) -> float:
        if self.num_queried_tokens == 0:
            return 0.0
        return self.num_hit_tokens / self.num_queried_tokens

    # -- the thing you should show the interviewer --------------------------
    def assert_invariants(self) -> None:
        pool = self.pool

        # 1. ref_count == 0  <=>  in the free queue
        queued: set[int] = set()
        node = pool.free_queue._head.next_free
        while node is not pool.free_queue._tail:
            assert node is not None
            assert node.ref_count == 0, f"block {node.block_id} idle but held"
            assert node.block_id not in queued, "block queued twice"
            queued.add(node.block_id)
            node = node.next_free
        for block in pool.blocks:
            assert (block.block_id in queued) == (block.ref_count == 0), (
                f"block {block.block_id}: ref={block.ref_count}, queued="
                f"{block.block_id in queued}"
            )

        # 2. every held reference corresponds to exactly one block-table slot
        held = sum(len(r.block_table) for r in self.requests.values())
        assert held == sum(b.ref_count for b in pool.blocks), "ref leak"

        # 3. the hash map only points at blocks that agree about their key
        for key, block in pool.hash_to_block.items():
            assert block.block_hash == key, "stale hash map entry"


# ---------------------------------------------------------------------------
# demo: four scenarios that together cover everything above
# ---------------------------------------------------------------------------
def _show(mgr: KVCacheManager, req_id: str) -> str:
    req = mgr.requests[req_id]
    return (
        f"table={mgr.block_table(req_id)} computed={req.num_computed_tokens}"
        f"/{req.num_tokens}"
    )


def demo() -> None:
    def title(text: str) -> None:
        print(f"\n{'=' * 72}\n{text}\n{'=' * 72}")

    # -- 1. prefill then decode --------------------------------------------
    title("1. Prefill, then decode: the basic loop")
    mgr = KVCacheManager(num_blocks=8, block_size=4)
    mgr.add_request("r1", list(range(10)))
    print(f"  add r1 with 10 tokens, block_size=4")

    hit = mgr.reuse_prefix("r1")
    print(f"  reuse_prefix -> {hit} tokens reused (cold cache)")

    ok = mgr.allocate_slots("r1", 10)
    print(f"  allocate_slots(r1, 10) -> {ok}   {_show(mgr, 'r1')}")
    print(f"  cache_full_blocks -> {mgr.cache_full_blocks('r1')} blocks published")

    for step in range(1, 6):
        before = len(mgr.requests["r1"].block_table)
        mgr.requests["r1"].append_token(1000 + step)  # sampler produced a token
        ok = mgr.allocate_slots("r1", 1)             # reserve room to compute it
        after = len(mgr.requests["r1"].block_table)
        grew = " +1 block" if after > before else ""
        mgr.cache_full_blocks("r1")
        print(f"  decode step {step}: allocate_slots(r1, 1) -> {ok}{grew}"
              f"   {_show(mgr, 'r1')}")
    mgr.assert_invariants()
    print("  invariants hold")

    # -- 2. prefix sharing --------------------------------------------------
    title("2. Prefix sharing: two requests, one copy of the system prompt")
    mgr = KVCacheManager(num_blocks=8, block_size=4)
    system_prompt = [1, 2, 3, 4, 5, 6, 7, 8]

    mgr.add_request("a", system_prompt + [9, 10])
    mgr.reuse_prefix("a")
    mgr.allocate_slots("a", 10)
    mgr.cache_full_blocks("a")
    print(f"  a: {_show(mgr, 'a')}  free={mgr.pool.num_free_blocks}")

    mgr.add_request("b", system_prompt + [11, 12])
    hit = mgr.reuse_prefix("b")
    print(f"  b: reuse_prefix -> {hit} tokens reused (system prompt is hot)")
    mgr.allocate_slots("b", 2)
    mgr.cache_full_blocks("b")
    print(f"  b: {_show(mgr, 'b')}  free={mgr.pool.num_free_blocks}")

    shared = [blk.block_id for blk in mgr.requests["b"].block_table
              if blk.ref_count > 1]
    print(f"  blocks shared by a and b: {shared}  (ref_count={2})")
    print(f"  prefix hit rate: {mgr.prefix_hit_rate:.0%}")
    mgr.assert_invariants()

    # -- 3. out of memory -> the scheduler preempts -------------------------
    title("3. Out of memory: allocate_slots returns False, scheduler preempts")
    mgr = KVCacheManager(num_blocks=4, block_size=4)
    mgr.add_request("big", list(range(16)))
    mgr.reuse_prefix("big")
    mgr.allocate_slots("big", 16)
    mgr.cache_full_blocks("big")
    print(f"  big: {_show(mgr, 'big')}  free={mgr.pool.num_free_blocks} (pool full)")

    mgr.add_request("small", [1, 2, 3])
    mgr.reuse_prefix("small")
    ok = mgr.allocate_slots("small", 3)
    print(f"  allocate_slots(small, 3) -> {ok}   <-- backpressure")
    print("  only the SCHEDULER can preempt; the allocator just says no.")

    mgr.free("big")
    print(f"  free(big) -> free={mgr.pool.num_free_blocks}"
          f" (blocks stay cached, not wiped)")
    ok = mgr.allocate_slots("small", 3)
    print(f"  retry allocate_slots(small, 3) -> {ok}   {_show(mgr, 'small')}")
    mgr.assert_invariants()

    # -- 4. eviction, and its cost ------------------------------------------
    title("4. Eviction: one page gone, the whole tail is unreachable")
    mgr = KVCacheManager(num_blocks=3, block_size=4)
    p_tokens = [7, 7, 7, 7, 7, 7, 7, 7, 9]  # 3 blocks, 2 of them full
    mgr.add_request("p", p_tokens)
    mgr.reuse_prefix("p")
    mgr.allocate_slots("p", 9)
    mgr.cache_full_blocks("p")
    print(f"  p: {_show(mgr, 'p')}  cached={mgr.num_cached_blocks}")

    mgr.free("p")
    print(f"  free(p) -> free={mgr.pool.num_free_blocks},"
          f" cached={mgr.num_cached_blocks}  (data outlives the request)")

    mgr.add_request("q", [42, 42, 42, 42])  # one block, unrelated prompt
    mgr.reuse_prefix("q")
    mgr.allocate_slots("q", 4)
    print(f"  q takes 1 block -> evictions={mgr.pool.num_evictions}"
          f"  (the least-recently-freed one)")
    print(f"  block 1 still holds cached content: "
          f"{mgr.pool.blocks[1].block_hash is not None}")

    mgr.add_request("r", p_tokens[:8] + [3])  # same 8-token prefix as p
    hit = mgr.reuse_prefix("r")
    print(f"  r asks for p's prefix again: reuse_prefix -> {hit} tokens")
    print("  ^ 0. The chain is broken at block 0, so block 1 is unreachable")
    print("    even though its KV bytes were never overwritten. Chained")
    print("    hashing can orphan a tail block; it just gets evicted later.")
    mgr.assert_invariants()

    print("\nAll invariants hold in every scenario.\n")


if __name__ == "__main__":
    demo()
