
import torch

device = "cuda"

# Persistent GPU buffers
x = torch.zeros(1024, device=device)
y = torch.ones(1024, device=device)

def run():
    a = x + y
    b = a * 2
    return torch.relu(b)

# Warm up on a separate CUDA stream
stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())

with torch.cuda.stream(stream):
    for _ in range(3):
        run()

torch.cuda.current_stream().wait_stream(stream)

# Capture GPU operations
graph = torch.cuda.CUDAGraph()

with torch.cuda.graph(graph):
    output = run()

# Reuse graph with new data
for i in range(10):
    x.fill_(i)     # Update data in the same buffer
    graph.replay()  # Replay captured operations

# Synchronize before inspecting results
torch.cuda.synchronize()
print(output[:5])
