import torch

def apply_rope(x: torch.Tensor, positions: torch.Tensor, base=10000.0):
    # x: (B,H,T,d)
    # positions: (T,)

    B, H, T, Dh = x.shape
    original_dtype = x.dtype

    pair_indices = torch.arange(
        0, Dh, 2,
        device=x.device,
        dtype=torch.float32
    )

    # 2. Calculate one inverse frequency per pair.
    exponent = pair_indices / Dh
    frequencies = base ** exponent
    inv_freq = 1.0 / frequencies

    # 3. Put positions into a column: (T, 1)
    positions = positions.to(
        device=x.device,
        dtype=torch.float32
    )
    positions_column = positions.unsqueeze(1)

    # 4: Put the frequencies into a row: (1, Dh // 2)
    frequencies_row = inv_freq.unsqueeze(0)

    # 5. Broadcasting calculates every angle.
    # Result: (T, Dh // 2)
    angles = positions_column * frequencies_row

    # 6. Calculate sine and cosine.
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)

    # 7. Add dimensions for batch and head broadcasting.
    # Both shapes: (B, H, T, Dh // 2, 2)
    x_pairs = x.float().reshape(B, H, T, Dh //2, 2)

    # 9. 