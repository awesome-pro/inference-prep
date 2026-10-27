
import torch


def apply_rope_simple(x, positions, base=10000.0):
    # x:         (B, H, T, Dh)
    # positions: (T,)

    B, H, T, Dh = x.shape
    original_dtype = x.dtype

    # Step 1: Create the indices for each feature pair.
    pair_indices = torch.arange(
        0, Dh, 2,
        device=x.device,
        dtype=torch.float32,
    )

    # Step 2: Calculate one inverse frequency per pair.
    exponent = pair_indices / Dh
    frequencies = base ** exponent
    inv_freq = 1.0 / frequencies

    # Step 3: Put positions into a column: (T, 1).
    positions = positions.to(
        device=x.device,
        dtype=torch.float32,
    )
    positions_column = positions.unsqueeze(1)

    # Step 4: Put frequencies into a row: (1, Dh // 2).
    frequencies_row = inv_freq.unsqueeze(0)

    # Step 5: Broadcasting calculates every angle.
    # Result: (T, Dh // 2)
    angles = positions_column * frequencies_row

    # Step 6: Calculate sine and cosine.
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)

    # Step 7: Add dimensions for batch and head broadcasting.
    # Both shapes: (1, 1, T, Dh // 2)
    cos_angles = cos_angles.unsqueeze(0).unsqueeze(0)
    sin_angles = sin_angles.unsqueeze(0).unsqueeze(0)

    # Step 8: Convert to FP32 and group adjacent features into pairs.
    # Shape: (B, H, T, Dh // 2, 2)
    x_pairs = x.float().reshape(B, H, T, Dh // 2, 2)

    # Step 9: Extract the first and second value of each pair.
    # Both shapes: (B, H, T, Dh // 2)
    x_first = x_pairs[..., 0]
    x_second = x_pairs[..., 1]

    # Step 10: Rotate every pair.
    y_first = (
        x_first * cos_angles
        - x_second * sin_angles
    )

    y_second = (
        x_first * sin_angles
        + x_second * cos_angles
    )

    # Step 11: Put each rotated pair back together.
    y_pairs = torch.stack(
        [y_first, y_second],
        dim=-1,
    )

    # Step 12: Restore the original feature dimension and dtype.
    y = y_pairs.flatten(-2)
    y = y.to(original_dtype)

    return y
