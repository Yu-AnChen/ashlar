"""Reproduce the BioFormats concurrent-read corruption without the full pipeline.

Reads every tile serially once (ground truth), then reads the same tiles from
many threads across TWO BioformatsReader instances and reports any tile whose
bytes differ from ground truth. Mismatches == the race that makes tiles look
"misplaced" in parallel assembly.

Usage (in the pixi env):

    # current code (shared class-level lock) -> expect PASS
    pixi run --manifest-path /Users/yuanchen/projects/pixi-env/ashlar/pixi.toml \
        python .dev/repro_bioformats_race.py FILE_A.rcpnl FILE_B.rcpnl

    # simulate the old per-instance lock -> expect FAIL (proves the repro works)
    pixi run ... python .dev/repro_bioformats_race.py FILE_A.rcpnl FILE_B.rcpnl \
        --per-instance-locks

You may pass the same path twice. --channel/--max-series/--rounds/--workers tune
the stress level.
"""
import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ashlar import reg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+", help="One or more rcpnl/image paths (>=1).")
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--max-series", type=int, default=40,
                   help="Limit tiles per reader to keep it fast.")
    p.add_argument("--rounds", type=int, default=20,
                   help="How many times to re-read every tile under load.")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--per-instance-locks", action="store_true",
                   help="Give each reader its own lock (simulates the OLD, buggy "
                        "behavior to confirm this script actually reproduces it).")
    args = p.parse_args()

    readers = [reg.BioformatsReader(pp) for pp in args.paths]
    if args.per_instance_locks:
        for r in readers:
            r._lock = threading.Lock()
        print("Using PER-INSTANCE locks (simulating the old behavior).")
    else:
        shared = readers[0]._lock
        print("Using shared lock:",
              "all readers share one lock object"
              if all(r._lock is shared for r in readers)
              else "WARNING: readers do NOT share a lock!")

    # Ground truth: read every (reader, series) once, serially.
    truth = {}
    work = []
    for ri, r in enumerate(readers):
        n = min(r.metadata.num_images, args.max_series)
        for s in range(n):
            img = r.read(s, args.channel)
            truth[(ri, s)] = np.asarray(img).copy()
            work.append((ri, s))
    print(f"Ground truth: {len(work)} tiles across {len(readers)} reader(s).")

    mismatches = []
    lock = threading.Lock()

    def task(ri, s):
        img = np.asarray(readers[ri].read(s, args.channel))
        if not np.array_equal(img, truth[(ri, s)]):
            with lock:
                mismatches.append((ri, s))

    jobs = work * args.rounds
    np.random.default_rng(0).shuffle(jobs)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(lambda a: task(*a), jobs))
    dt = time.perf_counter() - t0

    print(f"\nRan {len(jobs)} concurrent reads in {dt:.1f}s "
          f"({args.workers} workers).")
    if mismatches:
        uniq = sorted(set(mismatches))
        print(f"FAIL: {len(mismatches)} corrupted reads across "
              f"{len(uniq)} distinct tiles. Examples (reader, series): {uniq[:10]}")
    else:
        print("PASS: every concurrent read matched ground truth.")


if __name__ == "__main__":
    main()
