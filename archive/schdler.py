from collections import deque
from dataclasses import dataclass, field

import torch


class Block:
    def __init__(self, block_id: int) -> None:
        self.block_id = block_id
        self.ref_count = 0
        self.hash: int | None = None

WAITING, RUNNING, PREEMPTED, FINISHED = "waiting", "running", "preempted", "finished"

@dataclass
class Request:
    req_id: str
    tokens: list[int]
    num_computed_tokens: int
    status: str = WAITING
    n_output: int = 0
    block_table: list[Block] = field(default_factory=list)

    @property
    def num_token(self) -> int:
        return len(self.tokens)

    @property
    def is_finished(self) -> bool:
        return self.status == FINISHED

class KVCacheManager:
    def __init__(self, num_blocks: int, block_size: int) -> None:
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]

        self.free_q = deque(range(num_blocks))

        self.hash_to_block: dict[int, Block] = {}

        self.requests: dict[str, Request] = {}

    def add_request(self, req_id: str, tokens: list[int]) -> Request:
        pass

    def reuse_prefix(self, req_id) -> int:
        pass

    def allocate_slots(self, req_id, num_new_tokens) -> bool:
        pass

    def cache_full_blocks(self, req_id) -> None:
        pass

    def free(self, req_id: str) -> None:
        pass

    def get_block_table(self, req_id: str):
        pass

@dataclass
class NewRequestData:
    req_id: str
    prompt_token_ids: list[int]
    block_ids: 