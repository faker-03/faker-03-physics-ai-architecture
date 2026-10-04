import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hermitian Hamiltonian
H = torch.tensor(
    [[1 + 0j, 2 + 1j],
     [2 - 1j, 3 + 0j]],
    dtype=torch.complex64,
    device=device
)

dt = 0.1

# Anti-Hermitian generator
K = -1j * H

# Quantum propagator
U = torch.linalg.matrix_exp(K * dt)

# Identity matrix
I = torch.eye(
    H.shape[0],
    dtype=torch.complex64,
    device=device
)

print("device =", device)

print("\nH =")
print(H)

print("\nHermitian H?")
print(torch.allclose(H, H.conj().T))

print("\nK = -iH:")
print(K)

print("\nAnti-Hermitian K?")
print(torch.allclose(K.conj().T, -K))

print("\nU = exp(K dt):")
print(U)

print("\nU^dagger U =")
print(U.conj().T @ U)

print("\nUnitary U?")
print(
    torch.allclose(
        U.conj().T @ U,
        I,
        atol=1e-5
    )
)