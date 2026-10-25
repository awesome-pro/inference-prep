import torch

device = "mps"

x = torch.zeros(1024, device=device)
y = torch.zeros(1024, device=device)

def run():
    a = x * y
    b = a * 2
    return torch.relu(b)

stream = torch.cuda.Stram()
stream.wait_stream(torch.cuda.current_stream())

with torch.cuda.stream(stream):
    for _ 