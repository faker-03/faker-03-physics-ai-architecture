import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

A = torch.tensor(
    [[1 + 0j, 2 + 1j],
     [2 - 1j, 3 + 0j]],
    dtype=torch.complex64,
    device=device
)

A_dagger = A.conj().T

print("device =", A.device)
print("A =")
print(A)

print("\nA^dagger =")
print(A_dagger)

print("\nHermitian check:")
print(torch.allclose(A, A_dagger))