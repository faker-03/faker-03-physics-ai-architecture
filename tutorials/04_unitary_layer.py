import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class UnitaryLayer(nn.Module):
    def __init__(self, dim, dt=0.1):
        super().__init__()

        self.dt = dt

        real = torch.randn(dim, dim)
        imag = torch.randn(dim, dim)

        M = real + 1j * imag

        self.M = nn.Parameter(
            M.to(torch.complex64)
        )

    def forward(self, psi):
        M = self.M

        # Anti-Hermitian generator
        K = M - M.conj().T

        # Unitary propagator
        U = torch.linalg.matrix_exp(K * self.dt)

        return U @ psi, U


dim = 2

layer = UnitaryLayer(dim).to(device)

psi = torch.tensor(
    [1 + 0j, 0 + 0j],
    dtype=torch.complex64,
    device=device
)

psi_new, U = layer(psi)

I = torch.eye(
    dim,
    dtype=torch.complex64,
    device=device
)

print("device =", device)

print("\nU =")
print(U)

print("\nU^dagger U =")
print(U.conj().T @ U)

print("\nUnitary?")
print(
    torch.allclose(
        U.conj().T @ U,
        I,
        atol=1e-5
    )
)

print("\npsi =")
print(psi)

print("\npsi_new =")
print(psi_new)

print("\nNorm before =")
print(torch.sum(torch.abs(psi)**2))

print("\nNorm after =")
print(torch.sum(torch.abs(psi_new)**2))