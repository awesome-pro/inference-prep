# KV Cache Allocator — Interview Playbook

Companion to [`kv_cache_manager.py`](kv_cache_manager.py). Read the code once,
then practice the whiteboard skeleton in §4 from memory.

---

## 1. The 30-second opener (say this before writing anything)

> "A request's KV cache grows with its output, so reserving `max_model_len`
> per request is gigabytes and doesn't scale. I'd page the cache: fixed blocks
> of B tokens, each request gets a block table mapping logical→physical blocks,
> and we allocate on demand. Two requests with the same prefix can share
> physical blocks via ref counting. The manager exposes exactly one decision to
> the scheduler — `allocate_slots(req, n) -> bool`. `False` means OOM and the
> scheduler decides who to preempt."

Then draw: `[req] -> block table -> [b3][b7][b1] -> GPU pages`, and say
"`ref_count` is the bridge between the two."

Why open this way: it tells the interviewer you know the *contract* before you
know the code. Most candidates start writing a `Block` class and stall.

---

## 2. Vocabulary — get these exact, it reads as seniority

| Term | Layer | Who notices |
|---|---|---|
| **Block / page** | allocator | nobody — it's the unit |
| **Block table** | manager | the GPU kernel (it's an int32 tensor) |
| **Eviction** | allocator | **nobody** — a cached block had ref_count 0, we reused it |
| **Preemption** | scheduler | **the victim request** — it loses all its KV and recomputes |
| **Internal fragmentation** | — | the last partial block in a request |
| **Prefix hit** | manager | the request — it skips N tokens of prefill |

The eviction/preemption distinction is the single best signal in this question.
Say verbatim: *"eviction is invisible because the block was already free;
preemption is visible because the request was running."*

---

## 3. Order of work, timeboxed for a 60-minute slot

| Min | Do this | Say this |
|---|---|---|
| 0–3 | restate problem, define `allocate_slots` contract | "let me nail the API first" |
| 3–10 | `Block`, free queue, ref counts, the invariant | "ref_count==0 ⟺ in free queue" |
| 10–25 | `allocate_slots` + `free`, **no caching yet** | "correct first, fast later" |
| 25–40 | prefix caching: chained hash, lookup, publish | "hash folds in the parent" |
| 40–50 | eviction (LRU head) + the wake-up path | "O(1) remove from the middle" |
| 50–60 | invariants test, complexity, follow-ups | "here's how I'd test it" |

If you're running out of time, **skip prefix caching, not the invariants.** A
correct allocator with great tests beats a clever one that double-frees.

---

## 4. Minimal skeleton (~50 lines — this is what you memorize)

Deliberately uses `deque` + O(n) `remove()`. Say out loud: *"this is O(n) per
wake-up; with a doubly linked list it's O(1), which is why vLLM has its own
queue class instead of a deque."* Narration like that beats silently writing
the optimal version.

```python
from collections import deque

class KVAlloc:
    def __init__(self, num_blocks: int, block_size: int):
        self.bs = block_size
        self.free = deque(range(num_blocks))   # head = evicted first (LRU)
        self.ref = [0] * num_blocks
        self.tbl = {}                          # req_id -> [block_id]
        self.key = {}                          # chained hash -> block_id
        self.bkey = {}                         # block_id -> chained hash

    # ---- routing table ------------------------------------------------
    def alloc(self, rid: str, n_new: int, n_computed: int) -> bool:
        need = -(-(n_computed + n_new) // self.bs) - len(self.tbl.get(rid, []))
        if need > len(self.free):
            return False                       # atomic: nothing mutated
        for _ in range(need):
            b = self.free.popleft()            # <-- eviction happens here
            k = self.bkey.pop(b, None)
            if k is not None:
                del self.key[k]                # content is about to be lost
            self.ref[b] = 1
            self.tbl.setdefault(rid, []).append(b)
        return True

    def free(self, rid: str) -> None:
        for b in self.tbl.pop(rid, []):
            self.ref[b] -= 1
            if self.ref[b] == 0:
                self.free.append(b)            # keeps bkey -> still cached

    # ---- prefix cache -------------------------------------------------
    def publish(self, rid: str, toks: list[int], n_computed: int) -> None:
        parent = None
        for i, b in enumerate(self.tbl.get(rid, [])):
            end = (i + 1) * self.bs
            if end > n_computed:
                break                          # only full blocks are cacheable
            k = hash((parent, tuple(toks[end - self.bs:end])))
            if b not in self.bkey:
                self.bkey[b] = k
                self.key.setdefault(k, b)
            parent = k                         # chain even on a hit
        # (also needs: a map from req to tokens; elided here)

    def reuse(self, rid: str, toks: list[int]) -> int:
        parent = n = 0
        parent = None
        while n + self.bs <= len(toks) - 1:    # keep 1 token to compute!
            k = hash((parent, tuple(toks[n:n + self.bs])))
            b = self.key.get(k)
            if b is None:
                break
            if self.ref[b] == 0:
                self.free.remove(b)            # wake-up; O(n) here
            self.ref[b] += 1
            self.tbl.setdefault(rid, []).append(b)
            parent, n = k, n + self.bs
        return n
```

---

## 5. Six invariants — state them out loud, they're the interview

1. `ref_count == 0` **⟺** the block is in the free queue.
2. `sum(ref_count) == sum(len(block_table))` over all live requests.
3. `num_blocks == held + queued` — nothing is ever lost.
4. Allocate is **atomic**: check capacity first, then commit. Never half-allocate.
5. Only a **full** block may be hashed (its KV must describe a reusable prefix).
6. Max prefix reused is `num_tokens - 1` — you need one token left to forward.

---

## 6. Complexity

| Operation | Cost |
|---|---|
| `allocate_slots(n)` | O(blocks added), amortized |
| `free(req)` | O(blocks held) |
| `reuse_prefix` | O(prompt / B) hash lookups |
| evict one block | O(1) |
| wake a cached block | O(1) with linked list, O(n) with deque |
| memory | O(num_blocks) |

---

## 7. The five bugs that sink candidates

1. **Caching the last block.** `while n + B <= len(toks)` instead of
   `<= len(toks) - 1`. Full reuse → nothing to compute → no logits.
   *(This is the bug in your `allocator_with_prefix_caching.py:62`.)*
2. **Forgetting to remove a hit block from the free list.** Then the next
   `allocate` hands the *same physical block* to a different request. Silent
   KV corruption — the worst possible bug here.
3. **Non-atomic failure.** Decrementing free counts, then raising/returning
   `False` halfway. The scheduler retries and now memory is wrong forever.
4. **Breaking the hash chain** on publish: `parent = k` must be assigned even
   when the block was already cached, or every later key in that sequence is
   garbage.
5. **Double free** — `release()` without `assert ref_count > 0`. A block goes
   into the free list twice and gets handed out twice.

---

## 8. Follow-ups they will ask (with answers)

**"What if the request is 100k tokens?"** — chunked prefill: the *scheduler*
grants a token budget per step (e.g. 2048), the allocator just reserves blocks
for the granted slice. None of this logic changes.

**"How do you keep the GPU block table in sync?"** — it's an int32 tensor per
request, appended each step; the worker resets rows for preempted requests.
`SchedulerOutput` carries only the *new* block ids per request.

**"How do you prevent a hash collision across models?"** — fold model id /
weight version into the hash (see `scope` in your other file). A stale
collision = reusing another model's KV = garbage output.

**"What about partial blocks shared under copy-on-write?"** — if two requests
share a block and one writes into a partial tail, copy the block first
(`ref_count > 1` → allocate a private copy). vLLM keeps the last block private
to dodge most of this.

**"Why not always use a big block size?"** — fewer table entries and better
kernel efficiency, but more internal fragmentation on the tail and coarser
prefix sharing. vLLM defaults to 16.

**"Where's the eviction policy?"** — the free queue *is* the policy: freeing
appends to the tail, allocating pops the head, so it's LRU by free-time.
LFU/FIFO are a one-line swap. Say "LRU approximates reuse likelihood".

**"What happens on preemption?"** — the scheduler calls `free(req)`;
`num_computed_tokens` resets to 0, so the request recomputes everything. The
allocator isn't involved in the decision at all.

**"How would you measure it?"** — prefix hit rate = hit tokens / queried
tokens (that property exists in the code), evictions counter, and
free-block watermark over time.

---

## 9. Your 5-minute test strategy

Write `assert_invariants()` and call it after every scenario. Say:
*"I'd rather assert the invariant than the output."*

```python
def assert_invariants(self):
    # 1. ref_count == 0  <=>  in free queue
    # 2. sum(ref_count) == sum(len(table))
    # 3. every hash-map entry agrees with block.block_hash
```

Then four scenarios, in this order (all implemented in the demo):

1. prefill then decode — block growth at the boundary
2. two requests sharing a system prompt — ref_count hits 2
3. pool full → `False` → preempt → retry succeeds
4. eviction → an orphaned tail block that can no longer be reached

---

## 10. What was wrong with the two earlier drafts

| File | Problem |
|---|---|
| [`allocator.py`](allocator.py) | line 18 `deque([KVBlock])` stores the *class*, not instances → `len(free)` is 1 → first `allocate` raises `MemoryError`. Also no ref counting, no cache, raises instead of returning `False`. |
| [`allocator_with_prefix_caching.py`](allocator_with_prefix_caching.py) | truncated at line 154 (`def assert_invariants` with no body) → won't parse. Logic is otherwise close; the real bug is line 62 reusing the last block. |

Good ideas worth keeping from them: `scope` in the hash, `_empty` separate from
`_evictable`, `mark_computed` as an explicit publish step, and `blake2b` for
cross-process keys.

---

## 11. Map back to vLLM (you just read this code)

| Here | vLLM |
|---|---|
| `BlockPool` | [`v1/core/block_pool.py`](../vllm/vllm/v1/core/block_pool.py) |
| `FreeBlockQueue` | `FreeKVCacheBlockQueue` in `v1/core/kv_cache_utils.py` |
| `KVCacheManager.allocate_slots` | [`v1/core/kv_cache_manager.py`](../vllm/vllm/v1/core/kv_cache_manager.py) |
| `reuse_prefix` | `get_computed_blocks()` |
| `cache_full_blocks` | `cache_blocks()` |
| who calls `free` on preempt | `Scheduler._preempt_request` |
