# Kimi 任务备份：YOLO 基准测试（2026-10-03）

## 来源

- 原始位置：`C:\Users\<你的用户名>\Documents\kimi\tasks\2026-10-03`
- 由 Kimi（月之暗面）生成的任务工作区，本仓库为归档备份。

## 任务内容

在本机测试 YOLOv8（n/s/m）与 CPU 的推理性能，涉及 CUDA 环境安装与 pip 依赖记录。

## 本仓库包含（脚本与记录）

- `bench_yolo.py`、`bench_cpu.py` — 基准测试脚本
- `pip_install.log`、`pip_cuda.log` 等 — 环境安装日志

## 未包含（约 4.7 GB，见 `EXCLUDED_FILES.csv`）

- `.venv/` 虚拟环境（约 4.6 GB）
- 模型权重 `yolov8*.pt`、CUDA 安装包 `cuda_installer.exe`、测试视频 `bench_1080p.mp4`

被排除的大文件清单保存在根目录 `EXCLUDED_FILES.csv`。
