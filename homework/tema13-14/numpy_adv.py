# Ex 1. Broadcasting w bias


import numpy as np

img = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])

bias = np.array([1, 2, 3])

result = img + bias

print("\nOriginal image:\n", img)
print("\nBias:\n", bias)
print("\nResult after applying bias:\n", result)
print("\n\n")



# ================================================



# Ex 2. Indexare logica



arr = np.array([10, 25, 37, 42, 19, 55, 33])

mask = arr > 35

filtered = arr[mask]

print("\nOriginal array:", arr)
print("\nMask:", mask)
print("\nValues over 35:", filtered)



# ================================================



# Ex 3. Functii matematice si agregari


# Part 1
x = np.linspace(0, 10, 100)

y = np.sin(x)

mean_y = np.mean(y)
std_y = np.std(y)

print("Vector x:", x)
print("\nSine of x:", y)
print("\nMean (intermediate):", mean_y)
print("Standard deviation:", std_y)


# Part 2
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

sum_axis0 = np.sum(matrix, axis=0)  # column-wise
sum_axis1 = np.sum(matrix, axis=1)  # row-wise

mean_axis0 = np.mean(matrix, axis=0)
mean_axis1 = np.mean(matrix, axis=1)

print("\nMatrix:\n", matrix)
print("\nSum along axis=0 (columns):", sum_axis0)
print("Sum along axis=1 (rows):", sum_axis1)
print("Mean along axis=0:", mean_axis0)
print("Mean along axis=1:", mean_axis1)



# ================================================



# Ex 4. Normalizare pe axa


m = np.array([
    [10, 20],
    [30, 40]
])

media = np.mean(m, axis=0)
std = np.std(m, axis=0)

norm = (m - media) / std

print("Matrix originală:\n", m)
print("\nMedia (pe coloane):", media)
print("Abaterea standard (pe coloane):", std)
print("\nMatrice normalizată:\n", norm)



# ================================================



# Ex 5. Big arrays and filtration


arr = np.random.randint(1, 20001, size=10_000_000)
0
mask = arr > 10000

filtered = arr[mask]

mean_filtered = np.mean(filtered)

print("Total elements:", arr.size)
print("Elements > 10000:", filtered.size)
print("Intermediate (mean) of filtered values:", mean_filtered)



# ================================================



# Ex 6. Practical exercises


# Generează 1 milion de numere aleatoare și calculează suma celor divizibile cu 7.


arr = np.random.randint(1, 1_000_001, size=1_000_000)

mask = arr % 7 == 0

divisible_by_7 = arr[mask]

sum_divisible = np.sum(divisible_by_7)

print("Total numbers:", arr.size)
print("Numbers divisible by 7:", divisible_by_7.size)
print("Sum of numbers divisible by 7:", sum_divisible)