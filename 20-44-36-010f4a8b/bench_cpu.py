"""CPU 基准：单核 + 全核浮点计算。"""
import time
import math
from concurrent.futures import ProcessPoolExecutor


def work(n):
    x = 0.0
    for i in range(n):
        x += math.sqrt(i) * math.sin(i)
    return x


if __name__ == "__main__":
    # 单核
    work(500_000)  # 预热
    t0 = time.time()
    work(3_000_000)
    s = time.time() - t0
    print(f"single-core : {s:.2f}s")

    # 全核（12 线程）
    N = 12
    with ProcessPoolExecutor(N) as ex:
        list(ex.map(work, [100_000] * N))
        t0 = time.time()
        list(ex.map(work, [3_000_000] * N))
        m = time.time() - t0
    print(f"multi-core  : {m:.2f}s  (speedup {N * s / m:.1f}x)")
