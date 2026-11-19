"""Download complete image/mask pairs from a pinned public WalkScapes revision."""

from __future__ import annotations

import argparse
import hashlib
import random
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from walkscapes import SPLITS, WORK, sample_key


REPO = "https://github.com/KNSyNAPS-org/walkscapes-dataset-data.git"
REVISION = "938f6ab86028e51a66cc4196a432c055e1fb981c"
RAW = "https://raw.githubusercontent.com/KNSyNAPS-org/walkscapes-dataset-data"
THREAD = threading.local()


def download(entry: tuple[str, str], output: Path) -> None:
    relative, expected_sha = entry
    target = output / relative
    if target.is_file():
        return
    if not hasattr(THREAD, "session"):
        THREAD.session = requests.Session()
    for attempt in range(3):
        try:
            response = THREAD.session.get(f"{RAW}/{REVISION}/{relative}", timeout=(15, 45))
            response.raise_for_status()
            data = response.content
            git_blob = f"blob {len(data)}\0".encode() + data
            if hashlib.sha1(git_blob).hexdigest() != expected_sha:
                raise ValueError(f"Downloaded file differs from pinned Git blob: {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_suffix(target.suffix + ".part")
            temp.write_bytes(data)
            temp.replace(target)
            return
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=WORK / "walkscapes-index")
    parser.add_argument("--output", type=Path, default=WORK / "walkscapes-dataset-data")
    parser.add_argument("--limit", type=int, help="download this many pairs per official split")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if not (args.index / ".git").exists():
        args.index.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "git", "-c", "http.version=HTTP/1.1", "clone", "--depth", "1",
            "--filter=blob:none", "--no-checkout", REPO, str(args.index),
        ], check=True)
    git = ["git", "-C", str(args.index)]
    if subprocess.run(
        [*git, "cat-file", "-e", f"{REVISION}^{{commit}}"], capture_output=True,
    ).returncode:
        subprocess.run([
            *git, "-c", "http.version=HTTP/1.1", "fetch", "--depth", "1",
            "--filter=blob:none", "origin", REVISION,
        ], check=True)
    tree = subprocess.check_output([*git, "ls-tree", "-r", REVISION], text=True)
    blobs = {}
    for line in tree.splitlines():
        metadata, path = line.split("\t", 1)
        blobs[path] = metadata.split()[2]
    entries = [(name, blobs[name]) for name in ("LICENSE", "TERMS_OF_USE.md", "README.md")]
    for split in SPLITS:
        images = sorted(path for path in blobs if path.startswith(f"dataset/imgs/{split}/"))
        masks = {sample_key(Path(path)): path for path in blobs
                 if path.startswith(f"dataset/masks/{split}/")}
        if args.limit is not None:
            images = sorted(random.Random(42).sample(images, min(args.limit, len(images))))
        for image in images:
            mask = masks[sample_key(Path(image))]
            entries.extend(((image, blobs[image]), (mask, blobs[mask])))
        print(f"{split}: {len(images)} pairs", flush=True)
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        tasks = {pool.submit(download, item, args.output): item[0] for item in entries}
        for done, task in enumerate(as_completed(tasks), 1):
            try:
                task.result()
            except (requests.RequestException, ValueError, OSError) as error:
                failures.append(tasks[task])
                print(f"Failed: {error}", flush=True)
            if done % 20 == 0 or done == len(tasks):
                print(f"{done - len(failures)}/{len(tasks)} files ready", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} files unavailable. Re-run to resume.")


if __name__ == "__main__":
    main()
