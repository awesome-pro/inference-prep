import torch


def variable_length_batched_attention(q, k, v, length):
    B, H, T, Dh = q.shape

    # calculate the attention scores
    scores = (q.float() @ k.float().transpose(-2, -1)) ** (Dh ** -0.5)

    #2. find valid token positions
    positions = torch.arange(T, device=q.device)

    valid = positions[None, :] < length[None, :]

    # 3. seperate query and key validity
    query_valid = valid[:, None, :, None]
    key_valid = valid[:, None, None, :]

    # 4. A quey can attend only to the valid queried and keys
    allowed = query_valid & key_valid

    mask = positions[:, None] >= positions[None, :]
    allowed = allowed & mask

    scores = scores.masked_fill(~allowed, float("-inf"))

    has_valid_key = allowed.any(dim=-1, keepdim=True)

    scores = torch.where(has_valid_key, scores, torch.zeros_like(scores))

    # set invalid attention probs to zero
    attention = torch.softmax(scores, dim=-1)
    attention = attention.masked_fill(~allowed, 0.0)

    output = attention @ v.float()

    return output.to(v.dtype)
