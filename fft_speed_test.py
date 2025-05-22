
import numpy as np
import time
import matplotlib.pyplot as plt

def fft(x):
    N = len(x)
    if N <= 1:
        return x
    
    even_part = fft(x[0::2])
    odd_part = fft(x[1::2])
    
    factor = np.exp(-2j * np.pi * np.arange(N) / N)
    return np.concatenate([
        even_part + factor[:N//2] * odd_part,
        even_part + factor[N//2:] * odd_part
    ])

# Mal optimisé !!
# def fft_iterative(x):
#     x = np.asarray(x, dtype=complex)
#     N = x.shape[0]
#     levels = int(np.log2(N))
#     if 2**levels != N:
#         raise ValueError("size of x must be a power of 2")
    
#     # Bit-reversal permutation
#     bit_rev_indices = np.array([int('{:0{width}b}'.format(i, width=levels)[::-1], 2) for i in range(N)])
#     x = x[bit_rev_indices]
    
#     # Iterative FFT
#     size = 2
#     while size <= N:
#         half_size = size // 2
#         for i in range(0, N, size):
#             for j in range(half_size):
#                 k = j
#                 t = np.exp(-2j * np.pi * k / size) * x[i + j + half_size]
#                 x[i + j + half_size] = x[i + j] - t
#                 x[i + j] = x[i + j] + t
#         size *= 2
#     return x

def fft_iterative(x):
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    levels = int(np.log2(N))
    if 2**levels != N:
        raise ValueError("size of x must be a power of 2")
    
    bit_rev_indices = np.array([int('{:0{width}b}'.format(i, width=levels)[::-1], 2) for i in range(N)])
    x = x[bit_rev_indices]

    size = 2
    while size <= N:
        half_size = size // 2
        twiddles = np.exp(-2j * np.pi * np.arange(half_size) / size)
        for i in range(0, N, size):
            for j in range(half_size):
                t = twiddles[j] * x[i + j + half_size]
                x[i + j + half_size] = x[i + j] - t
                x[i + j] = x[i + j] + t
        size *= 2
    return x


def fft_iterative_optimized(x):
    x = np.asarray(x, dtype=np.complex128)
    N = x.shape[0]
    levels = int(np.log2(N))
    if 2**levels != N:
        raise ValueError("size of x must be a power of 2")
    
    # Bit-reversal permutation
    bit_rev = np.arange(N, dtype=int)
    bit_rev = bit_rev.reshape((-1, 1))
    rev = np.unpackbits(bit_rev.view(np.uint8), axis=1, bitorder='little')[:, -levels:]
    bit_rev_indices = rev.dot(1 << np.arange(levels))
    x = x[bit_rev_indices]

    # Iterative Cooley-Tukey FFT
    size = 2
    while size <= N:
        half_size = size // 2
        twiddles = np.exp(-2j * np.pi * np.arange(half_size) / size)
        for i in range(0, N, size):
            temp = twiddles * x[i + half_size:i + size]
            x[i + half_size:i + size] = x[i:i + half_size] - temp
            x[i:i + half_size] += temp
        size *= 2
    return x

def fft_numpy(x):
    return np.fft.fft(x)

N = 2**14
x = np.random.random(N)

# Measure performance for different input sizes
sizes = [2**i for i in range(8, 15)]  # 256 to 16384
recursive_times = []
iterative_times = []
numpy_times = []
iterative_times_optimized = []

for N in sizes:
    x = np.random.random(N)
    
    # Recursive FFT
    start = time.time()
    fft(x)
    end = time.time()
    recursive_times.append(end - start)
    
    # Iterative FFT
    start = time.time()
    fft_iterative(x)
    end = time.time()
    iterative_times.append(end - start)
    
    # NumPy FFT
    start = time.time()
    np.fft.fft(x)
    end = time.time()
    numpy_times.append(end - start)
    
    start = time.time()
    fft_iterative_optimized(x)
    end = time.time()
    iterative_times_optimized.append(end - start)

# Create speed comparison plot
plt.figure(figsize=(10, 6))
plt.plot(sizes, recursive_times, 'o-', label=f'Recursive FFT: {recursive_times[-1]:.06f}s', linewidth=2)
plt.plot(sizes, iterative_times, 's-', label=f'Iterative FFT: {iterative_times[-1]:.06f}s', linewidth=2)
plt.plot(sizes, iterative_times_optimized, 'x-', label=f'Optimized Iterative FFT: {iterative_times_optimized[-1]:.06f}s', linewidth=2)
plt.plot(sizes, numpy_times, '^-', label=f'NumPy FFT: {numpy_times[-1]:.06f}s', linewidth=2)
plt.xlabel('Input Size (N)')
plt.ylabel('Execution Time (seconds)')
plt.title('FFT Speed Comparison for an input of 16384 samples')
plt.legend()
plt.grid(True)
plt.show()
