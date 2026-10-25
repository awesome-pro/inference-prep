"""
paged_kv_torch.py

The same paged KV cache, but with real torch tensors.
Read after scheduler.py.

What changes when you use tensors
---------------------------------
Nothing about the logic. Paging is address translation, and translation is
integer arithmetic. What changes is that the arithmetic now produces tensors
the GPU kernel reads directly:

  * the block table becomes an int32 tensor        [num_reqs, max_blocks]
  * every scheduled token gets a *slot*:
        slot = block_id * block_size + offset_in_block
    and the batch carries a `slot_mapping` tensor   [num_tokens]
  * the KV cache is ONE tensor                    [num_blocks, block_size, 2, head_dim]

Writing this step's new K/V becomes a single scatter:

    cache.view(-1, 2, head_dim)[slot_mapping] = k_new

Reading a request's whole history becomes a single gather:

    cache[that_request_block_table].reshape(-1, 2, head_dim)

That is the entire mechanism. `block_size` is the page size; the block table
is a page table; `slot_mapping` is how the kernel finds the page for each
new token. Everything else is the code you already wrote.
"""

from collections import deque
from dataclasses import dataclass, field

import torch

HEAD_DIM = 8


# ======================================================================
# 1. the allocator -- unchanged from scheduler.py
# ======================================================================
class Block:
    __slots__ = ("block_id", "ref_count", "hash")

    def __init__(self, block_id: int) -> None:
        self.block_id = block_id
        self.ref_count = 0
        self.hash: int | None = None


@dataclass
class Request:
    req_id: str
    tokens: list[int]
    table: list[Block] = field(default_factory=list)
    computed: int = 0

    @property
    def num_tokens(self) -> int:
        return len(self.tokens)


class PagedKVCache:
    """Blocks + prefix cache + the actual tensor holding the K/V values."""

    def __init__(self, num_blocks: int, block_size: int, head_dim: int = HEAD_DIM):
        self.block_size = block_size
        self.head_dim = head_dim

        # THE memory. vLLM layers it as [blocks, block_size, 2, heads, dim];
        # one head here so you can print it.
        self.cache = torch.zeros(num_blocks, block_size, 2, head_dim)

        self.blocks = [Block(i) for i in range(num_blocks)]
        self.free: deque[int] = deque(range(num_blocks))
        self.cached: dict[int, Block] = {}

    def alloc(self, req: Request, n: int) -> bool:
        """Reserve blocks for n more tokens. False = nothing changed."""
        need = -(-(req.computed + n) // self.block_size) - len(req.table)
        if need > len(self.free):
            return False
        for _ in range(need):
            block = self.blocks[self.free.popleft()]
            if block.hash is not None:  # evicted: its KV is now garbage
                del self.cached[block.hash]
                block.hash = None
            block.ref_count = 1
            req.table.append(block)
        req.computed += n
        return True

    def drop(self, req: Request) -> None:
        for block in req.table:
            block.ref_count -= 1
            if block.ref_count == 0:
                self.free.append(block.block_id)
        req.table.clear()
        req.computed = 0

    def reuse(self, req: Request) -> int:
        parent = None
        reused = 0
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
            parent = key


# ======================================================================
# 2. the batch -- what the model actually receives
# ======================================================================
@dataclass
class PlanItem:
    req: Request
    start: int
    num_tokens: int


@dataclass
class Batch:
    input_ids: torch.Tensor      # [T]     token ids to run
    positions: torch.Tensor      # [T]     absolute position of each
    slot_mapping: torch.Tensor   # [T]     where each new K/V is written
    block_tables: torch.Tensor   # [R, B]  where each request's history lives
    seq_lens: torch.Tensor       # [R]     how much history each request has
    req_ids: list[str]


def build_batch(store: PagedKVCache, items: list[PlanItem]) -> Batch:
    """Turn the scheduler's plan into the tensors a kernel can consume."""
    input_ids: list[int] = []
    positions: list[int] = []
    slots: list[int] = []
    tables: list[list[int]] = []
    seq_lens: list[int] = []

    for item in items:
        req, start, n = item.req, item.start, item.num_tokens
        input_ids.extend(req.tokens[start : start + n])
        positions.extend(range(start, start + n))

        # THIS is the whole trick. A token at position p lives in
        #   block = table[p // block_size], offset = p % block_size
        # so its physical address is block_id * block_size + offset.
        for p in range(start, start + n):
            block_id = req.table[p // store.block_size].block_id
            slots.append(block_id * store.block_size + p % store.block_size)

        tables.append([b.block_id for b in req.table])
        seq_lens.append(req.computed)

    max_blocks = max((len(t) for t in tables), default=0)
    padded = torch.zeros(len(tables), max_blocks, dtype=torch.int32)
    for row, table in enumerate(tables):
        padded[row, : len(table)] = torch.tensor(table, dtype=torch.int32)

    return Batch(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        positions=torch.tensor(positions, dtype=torch.long),
        slot_mapping=torch.tensor(slots, dtype=torch.long),
        block_tables=padded,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        req_ids=[i.req.req_id for i in items],
    )


# ======================================================================
# 3. moving the data
# ======================================================================
def write_kv(store: PagedKVCache, batch: Batch, k_new, v_new) -> None:
    """Scatter this step's K/V into the paged cache. One line, no loops.

    This is what vLLM's `reshape_and_cache` kernel does. The kernel does it in
    parallel instead of one big index assignment, but the addressing is
    identical.
    """
    flat = store.cache.view(-1, 2, store.head_dim)  # [num_slots, 2, head_dim]
    flat[batch.slot_mapping, 0] = k_new
    flat[batch.slot_mapping, 1] = v_new


def read_kv(store: PagedKVCache, req: Request):
    """Gather one request's whole history back into contiguous order.

    [n_blocks, block_size, 2, head_dim] -> [computed, 2, head_dim]

    The attention kernel never materialises this; it walks the block table
    inside the kernel. We build it to prove the addressing is correct.
    """
    table = torch.tensor([b.block_id for b in req.table], dtype=torch.long)
    slots = store.cache[table].reshape(-1, 2, store.head_dim)
    history = slots[: req.computed]
    return history[:, 0], history[:, 1]


def attend(q, k, v):
    """Naive single-query attention over a gathered history."""
    scores = k @ q / (q.numel() ** 0.5)
    return torch.softmax(scores, dim=-1) @ v


# ======================================================================
# demo
# ======================================================================
def kv_for(tokens: list[int]):
    """Stand-in for a model's K/V projection: deterministic per token id."""
    base = torch.arange(1, HEAD_DIM + 1, dtype=torch.float32)
    k = torch.stack([base + t for t in tokens])
    v = torch.stack([base * 0.5 + t for t in tokens])
    return k, v


def admit(store: PagedKVCache, req: Request):
    """Prefix lookup -> allocate -> build batch -> write KV -> publish."""
    hit = store.reuse(req)
    start = req.computed
    n = req.num_tokens - start

    assert store.alloc(req, n), "should have fit"
    batch = build_batch(store, [PlanItem(req, start, n)])

    k, v = kv_for(req.tokens[start : start + n])
    write_kv(store, batch, k, v)
    store.publish(req)
    return hit, batch


def check(store: PagedKVCache, req: Request) -> None:
    """The test that matters: paged gather == contiguous reference."""
    k, v = read_kv(store, req)
    want_k, want_v = kv_for(req.tokens[: req.computed])
    assert torch.equal(k, want_k), "K gathered wrong"
    assert torch.equal(v, want_v), "V gathered wrong"


def main() -> None:
    store = PagedKVCache(num_blocks=6, block_size=4)

    # Two requests that share a 4-token system prompt.
    r0 = Request("r0", [11, 12, 13, 14, 21, 22])
    r1 = Request("r1", [11, 12, 13, 14, 31, 32])

    print("r0: 6 tokens = blocks [0, 1]")
    _, b0 = admit(store, r0)
    print(f"  input_ids    {b0.input_ids.tolist()}")
    print(f"  positions    {b0.positions.tolist()}")
    print(f"  slot_mapping {b0.slot_mapping.tolist()}  <- block*4 + offset")
    print(f"  block_tables {b0.block_tables.tolist()}")
    check(store, r0)

    print("\nr1: same first 4 tokens, different tail")
    hit, b1 = admit(store, r1)
    print(f"  prefix hit   {hit} tokens")
    print(f"  positions    {b1.positions.tolist()}")
    print(f"  slot_mapping {b1.slot_mapping.tolist()}")
    print(f"  block_tables {b1.block_tables.tolist()}")
    check(store, r1)

    print("\n  r0 blocks:", [b.block_id for b in r0.table])
    print("  r1 blocks:", [b.block_id for b in r1.table])
    print(f"  ref_count of shared block: {r0.table[0].ref_count}")

    # One decode step for r0: a new token at position 6.
    r0.tokens.append(777)
    start = r0.computed
    assert store.alloc(r0, 1)
    batch = build_batch(store, [PlanItem(r0, start, 1)])
    k, v = kv_for([777])
    write_kv(store, batch, k, v)
    check(store, r0)
    print(f"\nr0 decode: position {start} -> slot {batch.slot_mapping.tolist()}"
          f" (block 1, offset {start % store.block_size})")

    # The payoff: attention over the gathered history.
    # (q here is the token's own K, so it attends mostly to itself. A real
    # model projects q through a different weight matrix -- the point is only
    # that a gathered paged history is usable.)
    k_all, v_all = read_kv(store, r0)
    out = attend(k[-1], k_all, v_all)
    print(f"  gathered history {tuple(k_all.shape)} -> attention out "
          f"{tuple(out.shape)}")
    print(f"  {out.tolist()[:4]} ...")

    print("\npaged gather matched the contiguous reference for both requests.")
    print("The data structure did not change. Only its address format did.")


if __name__ == "__main__":
    main()
