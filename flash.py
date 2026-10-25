import math

scores = [1, 2, 3, 4]
values = [10, 20, 30, 40]

tile_size = 2

m = float("-inf")
l = 0
a = 0

for start in range(0, len(scores), tile_size):
    s_tile = scores[start: start + tile_size]
    v_tile = values[start: start + tile_size]

    m_new = max(m, max(s_tile))
    alpha = math.exp(m - m_new)

    p  = [math.exp(s - m_new) for s in s_tile]

    l = alpha * l + sum(p)
    a = alpha * a + sum(
        weight * value
        for weight, value in zip(p, v_tile)
    )

    m = m_new

    print(f"Tile ending at {start + len(s_tile)}")
    print(f" m={m:.4f}, l={l:.4f}, a={a:.4f}")

output = a/l

print(f"Final attention output: {output:.4f}")
