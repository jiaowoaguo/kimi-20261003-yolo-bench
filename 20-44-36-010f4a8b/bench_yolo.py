"""YOLO 视频分析基准测试：生成合成 1080p 视频，测试端到端 FPS 与纯推理 FPS。"""
import time
import cv2
import numpy as np
from ultralytics import YOLO

VIDEO = "bench_1080p.mp4"
W, H, FPS0, SECS = 1920, 1080, 30, 8

# ---- 1. 生成合成视频（运动目标，模拟监控画面）----
writer = cv2.VideoWriter(VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), FPS0, (W, H))
rng = np.random.default_rng(42)
objs = [(rng.integers(100, W - 100), rng.integers(100, H - 100),
         rng.integers(-5, 6), rng.integers(-4, 5)) for _ in range(12)]
for f in range(FPS0 * SECS):
    frame = np.full((H, W, 3), 32, np.uint8)
    for i, (x, y, dx, dy) in enumerate(objs):
        nx, ny = x + dx * f, y + dy * f
        nx = max(40, min(W - 40, nx)); ny = max(40, min(H - 40, ny))
        cv2.rectangle(frame, (nx - 35, ny - 35), (nx + 35, ny + 35), (80, 160, 255), -1)
    cv2.putText(frame, f"frame {f}", (30, 50), 0, 1, (255, 255, 255), 2)
    writer.write(frame)
writer.release()
print(f"test video: {VIDEO}  {W}x{H} @{FPS0}fps, {FPS0*SECS} frames")

results = []
for mname in ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]:
    model = YOLO(mname)
    model.to("cuda")

    # 预热
    for _ in model.predict(VIDEO, stream=True, verbose=False, device=0):
        break

    # 端到端（解码+预处理+推理+后处理）
    t0 = time.time(); n = 0
    for _ in model.predict(VIDEO, stream=True, verbose=False, device=0):
        n += 1
    e2e = n / (time.time() - t0)

    # 纯推理（640x640 批量，GPU 已预热）
    batch = [np.zeros((640, 640, 3), np.uint8) for _ in range(16)]
    for _ in range(3):
        model.predict(batch, verbose=False, device=0)
    t0 = time.time(); reps = 10
    for _ in range(reps):
        model.predict(batch, verbose=False, device=0)
    infer = 16 * reps / (time.time() - t0)

    mem = __import__("torch").cuda.max_memory_allocated() / 2**30
    results.append((mname, e2e, infer, mem))
    print(f"{mname}: end-to-end {e2e:.1f} FPS | batch inference {infer:.1f} FPS | VRAM {mem:.2f} GB")

print("\n=== summary (RTX 3070, CUDA 12.6) ===")
for m, e, i, v in results:
    print(f"{m:14s} e2e={e:6.1f} FPS  infer={i:6.1f} FPS  vram={v:.2f} GB")
