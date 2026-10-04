import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.tensor(2.0, requires_grad=True, device=device)

y = x**3

y.backward()

print("device =", x.device)
print("x =", x.item())
print("y =", y.item())
print("dy/dx =", x.grad.item())