import numpy as np
import time
import tracemalloc
import requests
from io import BytesIO
from PIL import Image
from numba import jit, prange

def profile(func, *args, label="Function"):
    tracemalloc.start()
    start = time.perf_counter()

    result = func(*args)

    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "label": label,
        "time": elapsed,
        "memory": peak / 1024 / 1024,
        "result": result
    }
print("Loading data...")
url1 = "http://getwallpapers.com/wallpaper/full/c/a/7/1235918-3000-x-3000-hd-wallpapers-3000x2000-for-hd-1080p.jpg"
url2 = "http://getwallpapers.com/wallpaper/full/c/4/1/1235927-3000-x-3000-hd-wallpapers-3000x2000-screen.jpg"

try:
    A_raw = np.asarray(Image.open(BytesIO(requests.get(url1).content))) / 255.0
    B_raw = np.asarray(Image.open(BytesIO(requests.get(url2).content))) / 255.0
    A = np.mean(A_raw, axis=2) * 255
    B = np.mean(B_raw, axis=2) * 255
    C = np.random.rand(*A.shape) * 255
except Exception as e:
    print(f"Connection error: {e}. Using dummy data.")
    A, B, C = np.random.rand(2000, 3000), np.random.rand(2000, 3000), np.random.rand(2000, 3000)

def loop1_base(A, B):
    a = np.copy(A)
    b = np.copy(B)

    for j in range(1, a.shape[0] - 1):
        for i in range(1, a.shape[1] - 1):
            a[j, i] = a[j - 1, i] + b[j, i]
            b[j, i] = b[j, i] ** 2

    return a, b

@jit(nopython=True, parallel=False) 
def loop1_sol(A, B):
    a = A.astype(np.float32)
    b = B.astype(np.float32)
    
    rows = a.shape[0]
    cols = a.shape[1]

    for j in range(1, rows - 1):
        for i in range(1, cols - 1):
            a[j, i] = a[j - 1, i] + b[j, i]
            b[j, i] = b[j, i] ** 2
            
    return a, b
r_base = profile(loop1_base, A, B, label="Loop1 Base")
r_sol  = profile(loop1_sol, A, B, label="Loop1 Solution")

print("\n[Loop 1]")
print(f"Base Time: {r_base['time']:.4f}s | Mem: {r_base['memory']:.2f}MB")
print(f"Sol  Time: {r_sol['time']:.4f}s | Mem: {r_sol['memory']:.2f}MB")
print("Correct:",
      np.allclose(r_base["result"][0], r_sol["result"][0]) and
      np.allclose(r_base["result"][1], r_sol["result"][1]))

def loop2_base(A, B, C):
    a = np.copy(A)
    b = np.copy(B)

    for j in range(1, a.shape[0] - 1):
        for i in range(1, a.shape[1] - 1):
            a[j][i] = a[j - 1][i] + C[j][i]

        for i in range(1, b.shape[1] - 1):
            b[j][i] = b[j][i - 1] + C[j][i]

    return a, b

@jit(nopython=True)
def loop2_sol(A, B, C):
    a = A.astype(np.float32)
    b = B.astype(np.float32)
    c_local = C.astype(np.float32)
    
    rows, cols = a.shape

    for j in range(1, rows - 1):
        for i in range(1, cols - 1):
            a[j, i] = a[j - 1, i] + c_local[j, i]
        
        for i in range(1, cols - 1):
            b[j, i] = b[j, i - 1] + c_local[j, i]

    return a, b

r_base = profile(loop2_base, A, B, C, label="Loop2 Base")
r_sol  = profile(loop2_sol, A, B, C, label="Loop2 Solution")

print("\n[Loop 2]")
print(f"Base Time: {r_base['time']:.4f}s | Mem: {r_base['memory']:.2f}MB")
print(f"Sol  Time: {r_sol['time']:.4f}s | Mem: {r_sol['memory']:.2f}MB")
print("Correct:",
      np.allclose(r_base["result"][0], r_sol["result"][0]) and
      np.allclose(r_base["result"][1], r_sol["result"][1]))

Am = np.random.rand(100, 500)
Bm = np.random.rand(500, 250)
Dm = np.random.rand(100, 250)
alpha = 1.5
beta = 1.2

def sgemm_base(alpha, A, B, beta, D):
    d = np.copy(D)
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            d[i, j] *= beta
        for k in range(A.shape[1]):
            for j in range(d.shape[1]):
                d[i, j] += alpha * A[i, k] * B[k, j]
    return d

def sgemm_solution(alpha, A, B, beta, D):
    res = np.copy(D)
    res *= beta
    res += alpha * (np.matmul(A, B))
    
    return res

r_base = profile(sgemm_base, alpha, Am, Bm, beta, Dm, label="SGEMM Base")
r_sol  = profile(sgemm_solution, alpha, Am, Bm, beta, Dm, label="SGEMM Solution")

print("\n[SGEMM]")
print(f"Base Time: {r_base['time']:.4f}s | Mem: {r_base['memory']:.2f}MB")
print(f"Sol  Time: {r_sol['time']:.4f}s | Mem: {r_sol['memory']:.2f}MB")
print("Correct:", np.allclose(r_base["result"], r_sol["result"]))

def Laplacian_Operator_Base(A):
    x, y = A.shape
    laplacian = np.empty((x - 2, y - 2))

    for i in range(1, x - 1):
        for j in range(1, y - 1):
            laplacian[i-1, j-1] = (
                abs(A[i-1, j] + A[i+1, j] + A[i, j-1] + A[i, j+1] - 4*A[i, j]) > 0.05
            )
    return laplacian

@jit(nopython=True, parallel=True)
def Laplacian_Operator_Sol(A):
    x, y = A.shape
    laplacian = np.empty((x - 2, y - 2), dtype=np.bool_)
    
    for i in prange(1, x - 1):
        for j in range(1, y - 1):
            val = A[i-1, j] + A[i+1, j] + A[i, j-1] + A[i, j+1] - 4 * A[i, j]
            laplacian[i-1, j-1] = abs(val) > 0.05
            
    return laplacian
r_base = profile(Laplacian_Operator_Base, A, label="Laplacian Base")
r_sol  = profile(Laplacian_Operator_Sol, A, label="Laplacian Solution")

print("\n[Laplacian]")
print(f"Base Time: {r_base['time']:.4f}s | Mem: {r_base['memory']:.2f}MB")
print(f"Sol  Time: {r_sol['time']:.4f}s | Mem: {r_sol['memory']:.2f}MB")
print("Correct:", np.array_equal(r_base["result"], r_sol["result"]))