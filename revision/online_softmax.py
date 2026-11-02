import torch

def online_softmax(x: torch.Tensor, chunk_size: int = 2) -> torch.Tensor:
    # x dim - (M, N)
    x = x.float()

    R, C = x.shape

    m: torch.Tensor | None = None
    l: torch.Tensor | None = None
    start = 0

    for start in (0, C, chunk_size):
        end = min(start + chunk_size, C)

        chunk = x[:, start: end]
        chunk_max = chunk.max(dim=-1, keepdim=True).values

        if m is None:
            m = chunk_max
            l = torch.exp(chunk - m).sum(dim=-1, keepdim=True)
        else:
            m_new = torch.maximum(m, chunk_max)

            alpha = torch.exp(m - m_new)
            beta = torch.exp(chunk - m_new)

            l = alpha * l + beta.sum(dim=-1, keepdim=True)

            m = m_new

    output = torch.empty_like(x)

    for start in range(0, C, chunk_size):
        end = min(start + chunk_size, C)

        chunk = x[:, start: end]

        output[:, start:end] = torch.exp((chunk -m) / l)
        
    return output



