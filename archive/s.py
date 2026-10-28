"""Interview-sized vLLM V1-inspired continuous batching scheduler.

This is a scheduler simulator, not vLLM's real scheduler or a model runner.
It supports dynamic admissions, per-step token budgets, chunked prefill,
per-request KV-block accounting, and automatic completion + block release.
The request-output count is simulated (no token sampling/model execution).
"""
from collections import OrderedDict, deque
from dataclasses import dataclass


@dataclass
class Request:
    request_id: str
    prompt_len: int
    max_new_tokens: int
    computed: int = 0
    generated: int = 0
    blocks: int = 0

    @property
    def known(self) -> int:
        # Known tokens include prompt and already sampled output tokens.
        return self.prompt_len + self.generated

    @property
    def needed(self) -> int:
        return self.known - self.computed


class ContinuousBatchScheduler:
    def __init__(self, max_num_batched_tokens: int,
                 max_num_seqs: int, num_kv_blocks: int,
                 block_size: int = 4):
        if min(max_num_batched_tokens, max_num_seqs,
               num_kv_blocks, block_size) <= 0:
            raise ValueError("All scheduler capacities must be positive")
        self.token_limit = max_num_batched_tokens
        self.seq_limit = max_num_seqs
        self.block_size = block_size
        self.free_blocks = num_kv_blocks
        self.total_blocks = num_kv_blocks
        self.waiting = deque()
        self.running = OrderedDict()
        self.finished = []
        self.seen_ids = set()

    def add(self, request: Request) -> None:
        if request.request_id in self.seen_ids:
            raise ValueError("Duplicate request ID")
        if request.prompt_len <= 0 or request.max_new_tokens <= 0:
            raise ValueError("Prompt length and max_new_tokens must be positive")
        if request.computed or request.generated or request.blocks:
            raise ValueError("New requests must not have existing progress")
        self.seen_ids.add(request.request_id)
        self.waiting.append(request)

    def _fit_tokens(self, request: Request, wanted: int) -> int:
        # Existing partially filled KV blocks plus any free KV blocks.
        capacity_tokens = (
            (request.blocks + self.free_blocks) * self.block_size
        )
        return max(0, min(wanted, capacity_tokens - request.computed))

    def _reserve(self, request: Request, tokens: int) -> None:
        required_blocks = (
            (request.computed + tokens + self.block_size - 1)
            // self.block_size
        )
        extra = max(0, required_blocks - request.blocks)
        assert extra <= self.free_blocks
        self.free_blocks -= extra
        request.blocks += extra

    def _schedule(self):
        """Make one iteration's plan: {request_id: token_count}."""
        budget = self.token_limit
        plan = {}

        # Running requests get priority (typically decode or prefill chunks).
        for request in list(self.running.values()):
            if budget == 0:
                break
            n = self._fit_tokens(request, min(request.needed, budget))
            if n <= 0:
                continue
            self._reserve(request, n)
            plan[request.request_id] = n
            budget -= n

        # Admit waiting requests as slots and token/KV budgets permit.
        while self.waiting and len(self.running) < self.seq_limit and budget:
            request = self.waiting[0]
            n = self._fit_tokens(request, min(request.needed, budget))
            if n <= 0:
                break  # Avoid admitting a zero-work request.