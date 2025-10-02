#!/usr/bin/env python3
# multi_gpu_e2fgvi.py
# -*- coding: utf-8 -*-
"""
Multi-GPU E2FGVI-HQ runner (process version)
- Single tqdm bar, optional verbose subprocess output
- Ingests per-episode source/merged mask videos
- Runs multiple workers per GPU (even split)
"""
import argparse, os, sys, subprocess
from multiprocessing import Process, JoinableQueue, Queue
from typing import List, Tuple
from pathlib import Path

import cv2, numpy as np  # noqa: E402
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = ROOT / 'rovi-aug-extension'
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from config import config

DEFAULT_OUTPUT_ROOT = Path('/home/guanhuaji/test/oxe-aug/videos')

CKPT_PATH = "release_model/E2FGVI-HQ-CVPR22.pth"
# ───────────────────────── discover ──────────────────────────────────
def make_pair(ep: str, base_dir: Path, dilution: int) -> Tuple[str, str, str, int]:
    """Return (video_path, mask_path, save_path, dilution) for one episode."""
    ep_dir = base_dir / ep
    return (
        str(ep_dir / "source_video.mp4"),
        str(ep_dir / "merged_mask.mp4"),
        str(ep_dir / "inpainting.mp4"),
        dilution,
    )

def discover_range(start: int, end: int, base_dir: Path, dilution: int) -> List[Tuple[str, str, str, int]]:
    return [make_pair(str(i), base_dir, dilution)
            for i in range(start, end)]

# ───────────────────────── subprocess wrapper ────────────────────────
def run_subprocess(video_path, mask_path, save_path, dilution, row, suppress):
    cmd = [
        sys.executable, "demo.py",
        "--model",  "e2fgvi_hq",
        "--video",  video_path,
        "--mask",   mask_path,
        "--ckpt",   CKPT_PATH,
        "--save_frame", save_path,
        "--dilution", str(dilution),
    ]
    env = os.environ.copy()
    env["TQDM_POS"] = str(row)  # mostly irrelevant since we suppress stdout
    env["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

    # Silence child process
    stdout = subprocess.DEVNULL if suppress else None
    stderr = subprocess.STDOUT if suppress else None
    subprocess.run(cmd, check=True, env=env, stdout=stdout, stderr=stderr)

# ───────────────────────── episode driver ────────────────────────────
def run_episode(video_path, mask_path, save_path, dilution, row, suppress):
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"source video not found: {video_path}")
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(f"merged mask not found: {mask_path}")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    run_subprocess(video_path, mask_path, save_path, dilution, row, suppress)

# ───────────────────────── multi-GPU worker ──────────────────────────
def worker_proc(gpu_id: int, worker_slot: int, job_q: JoinableQueue, done_q: Queue, failed_file: str, suppress: bool) -> None:
    """Consumes jobs from `job_q` using GPU `gpu_id`.
       Always notifies `done_q` when a job finishes (success or fail)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cv2.setNumThreads(0)

    # Give each worker a stable row line
    row = gpu_id * 10 + worker_slot

    while True:
        item = job_q.get()
        if item is None:
            job_q.task_done()
            break

        video_path, mask_path, save_path, dilution = item
        try:
            run_episode(video_path, mask_path, save_path, dilution, row, suppress)
        except Exception as e:
            ep_id = Path(video_path).parent.name
            print(f"[GPU {gpu_id} W{worker_slot}] ❌ Episode {ep_id}: {e}")
            try:
                with open(failed_file, "a", encoding="utf-8") as ff:
                    ff.write(f"{ep_id}\n")
            except Exception as fe:
                print(f"[GPU {gpu_id} W{worker_slot}] ⚠️  Could not write to {failed_file}: {fe}")
        finally:
            done_q.put(1)   # notify main progress bar
            job_q.task_done()

# ───────────────────────── helpers ───────────────────────────────────
def filter_existing_jobs(jobs: List[Tuple[str, str, str, int]]) -> Tuple[List[Tuple[str, str, str, int]], int]:
    """Return (remaining_jobs, skipped_count) by checking if save_path exists."""
    remaining = []
    skipped = 0
    for job in jobs:
        save_path = job[2]
        if os.path.isfile(save_path):
            print(f"Skipping existing: {save_path}")
            skipped += 1
            continue
        remaining.append(job)
    return remaining, skipped

def split_workers_evenly(num_workers: int, gpus: List[int]) -> List[int]:
    """Evenly split num_workers across GPUs; earlier GPUs get +1 if remainder."""
    if num_workers < 1:
        raise ValueError("--num_workers must be >= 1")
    if len(gpus) < 1:
        raise ValueError("At least one GPU id must be provided")
    base = num_workers // len(gpus)
    rem  = num_workers %  len(gpus)
    return [base + (1 if i < rem else 0) for i in range(len(gpus))]

# ───────────────────────── main ──────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser("Multi-GPU E2FGVI-HQ runner (process version)")
    ap.add_argument("--dataset", required=True, help="Dataset name")
    ap.add_argument("--split", required=True, help="Dataset split (train/val/test)")
    ap.add_argument("--start", type=int, required=True, help="First episode id (inclusive)")
    ap.add_argument("--end", type=int, required=True, help="Last episode id (exclusive)")
    ap.add_argument("--gpus", nargs="+", type=int, default=[0], help="GPU IDs, e.g. --gpus 0 1 2 3")
    ap.add_argument("-v", "--verbose", action="store_true", help="Show subprocess output")
    ap.add_argument("--dilution", type=int, default=0, help="Mask dilation factor (default: 0)")
    ap.add_argument("--num_workers", type=int, default=None,
                    help="Total worker processes across all GPUs (evenly split). Default: len(--gpus)")
    args = ap.parse_args()

    suppress_output = not args.verbose

    if args.end <= args.start:
        sys.exit("--end must be greater than --start")

    dataset_cfg = config.get(args.dataset, {})
    if not dataset_cfg:
        sys.exit(f"Unknown dataset {args.dataset}")

    out_cfg = dataset_cfg.get('out_path')
    if out_cfg:
        out_root = Path(out_cfg)
        if not out_root.is_absolute():
            out_root = DEFAULT_OUTPUT_ROOT / out_root
    else:
        out_root = DEFAULT_OUTPUT_ROOT

    base_dir = out_root / args.dataset / args.split

    jobs = discover_range(args.start, args.end, base_dir, args.dilution)

    if not jobs:
        print("✅ Nothing to do. Exiting.")
        return

    failed_file = os.path.join('/tmp', 'failed_episodes.txt')
    Path('/tmp').mkdir(parents=True, exist_ok=True)
    open(failed_file, 'w').close()

    print(f"[INFO] Dispatching {len(jobs)} episode(s) across GPUs {args.gpus}")

    print(f"[INFO] Dispatching {len(jobs)} episode(s) across GPUs {args.gpus}")

    # ── queues ───────────────────────────────────────────────────────
    job_q:  JoinableQueue = JoinableQueue()
    done_q: Queue         = Queue()

    for j in jobs:
        job_q.put(j)

    # ── worker allocation ────────────────────────────────────────────
    if args.num_workers is None:
        args.num_workers = len(args.gpus)

    per_gpu_counts = split_workers_evenly(args.num_workers, args.gpus)
    total_workers = sum(per_gpu_counts)

    # one poison pill per worker
    for _ in range(total_workers):
        job_q.put(None)

    # ── launch workers ───────────────────────────────────────────────
    procs: List[Process] = []
    for gi, gid in enumerate(args.gpus):
        for w in range(per_gpu_counts[gi]):
            p = Process(target=worker_proc, args=(gid, w, job_q, done_q, failed_file, suppress_output))
            p.start()
            procs.append(p)

    # helpful log
    alloc_str = ", ".join(f"GPU{gid}:{per_gpu_counts[i]}" for i, gid in enumerate(args.gpus))
    print(f"[INFO] Worker allocation → {alloc_str} (total {total_workers})")

    # ── global progress bar ──────────────────────────────────────────
    with tqdm(total=len(jobs), desc="Episodes", unit="ep") as pbar:
        completed = 0
        while completed < len(jobs):
            done_q.get()       # blocks until a worker reports completion
            completed += 1
            pbar.update(1)

    # ── cleanup ──────────────────────────────────────────────────────
    job_q.join()
    for p in procs:
        p.join()

    print("✅ All episodes finished.")
    print(f"📄 Failed episodes (if any) recorded in: {failed_file}")

if __name__ == "__main__":
    main()





'''
Example:
python batched_inference_e2fgvi.py --dataset jaco_play --split train --start 0 --end 5 --gpus 0 1
'''
