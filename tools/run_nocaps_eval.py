import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import requests
try:
    import ruamel_yaml as yaml  # type: ignore[import]
except ModuleNotFoundError:  # pragma: no cover - optional dependency name
    try:
        from ruamel import yaml  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover - final fallback
        import yaml  # type: ignore
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import eval_nocaps  # type: ignore  # pylint: disable=wrong-import-position


_THREAD_LOCAL = threading.local()


def _get_session() -> requests.Session:
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        _THREAD_LOCAL.session = session
    return session


def fetch_image(url: str, dest: Path, retries: int, timeout: int) -> Tuple[bool, Optional[str]]:
    """Download an image to `dest`, retrying if needed."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: str | None = None
    for attempt in range(1, retries + 1):
        try:
            response = _get_session().get(url, timeout=timeout)
        except Exception as exc:  # pragma: no cover - network errors
            last_err = str(exc)
        else:
            if response.status_code == 200:
                dest.write_bytes(response.content)
                return True, None
            last_err = f"HTTP {response.status_code}"
        time.sleep(min(5, attempt))
    return False, last_err


def _index_openimages(root: Optional[Path]) -> Dict[str, Path]:
    """Build an index of Open Images files by ID (stem, case-insensitive)."""
    index: Dict[str, Path] = {}
    if root is None or not root.exists():
        return index
    for p in root.rglob("*.jp*g"):
        index[p.stem.lower()] = p.resolve()
    for p in root.rglob("*.png"):
        index[p.stem.lower()] = p.resolve()
    return index


def _try_copy(src: Optional[Path], dest: Path) -> bool:
    if src is None:
        return False
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copyfile(str(src), str(dest))
        return True
    except Exception:
        return False


def _detect_coco_split(file_name: str, url: Optional[str]) -> Optional[str]:
    s = (url or "") + " " + file_name
    if "train2014" in s:
        return "train2014"
    if "val2014" in s:
        return "val2014"
    if re.search(r"_train2014_", file_name):
        return "train2014"
    if re.search(r"_val2014_", file_name):
        return "val2014"
    return None


def _ensure_openimages_cli(pip_install: bool) -> bool:
    try:
        completed = subprocess.run(["openimages", "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if completed.returncode == 0:
            return True
    except Exception:
        pass
    if not pip_install:
        return False
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "openimages"], check=True)
        completed = subprocess.run(["openimages", "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return completed.returncode == 0
    except Exception:
        return False


def _download_openimages_id(image_id: str, tmp_dir: Path, timeout: int) -> Optional[Path]:
    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["openimages", "download", "--image", image_id, "--destination", str(tmp_dir)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout * 3 if timeout else None,
            check=False,
        )
        stem = image_id.lower()
        for ext in (".jpg", ".jpeg", ".png"):
            c1 = tmp_dir / f"{stem}{ext}"
            if c1.exists():
                return c1
        for p in tmp_dir.rglob("*"):
            if p.is_file() and p.stem.lower() == stem:
                return p
    except Exception:
        return None
    return None


def convert_split(
    raw_data: Dict,
    split: str,
    image_root: Path,
    retries: int,
    timeout: int,
    skip_download: bool,
    workers: int,
    coco_train_root: Optional[Path],
    coco_val_root: Optional[Path],
    openimages_index: Dict[str, Path],
    use_openimages_cli: bool,
    pip_install_openimages: bool,
) -> Tuple[List[Dict], List[Dict]]:
    """Create BLIP-ready annotations and download missing images for one split."""
    images = raw_data["images"]
    annotations = raw_data.get("annotations", [])
    id_to_captions: Dict[int, List[str]] = {}

    for ann in annotations:
        img_id = int(ann["image_id"])
        id_to_captions.setdefault(img_id, []).append(ann["caption"])

    converted: List[Dict] = []
    failures: List[Dict] = []

    _openimages_ready = None  # lazy init

    def process_image(img: Dict) -> Tuple[Optional[Dict], Optional[Dict]]:
        file_name = img["file_name"]
        url = img.get("coco_url") or img.get("flickr_url")
        openimg_id = img.get("open_images_id") or img.get("openimages_id")

        dest = image_root / split / file_name
        if dest.exists():
            pass
        else:
            # 1) Try local COCO mapping
            if url:
                coco_split = _detect_coco_split(file_name, url)
                src: Optional[Path] = None
                if coco_split == "train2014" and coco_train_root is not None:
                    c = coco_train_root / file_name
                    if c.exists():
                        src = c
                elif coco_split == "val2014" and coco_val_root is not None:
                    c = coco_val_root / file_name
                    if c.exists():
                        src = c
                if _try_copy(src, dest):
                    entry = {"image": f"{split}/{file_name}", "img_id": int(img["id"])}
                    if id_to_captions:
                        entry["captions"] = id_to_captions.get(int(img["id"]), [])
                    return entry, None

            # 2) Try Open Images index
            if openimg_id:
                src2 = openimages_index.get(str(openimg_id).lower())
                if _try_copy(src2, dest):
                    entry = {"image": f"{split}/{file_name}", "img_id": int(img["id"])}
                    if id_to_captions:
                        entry["captions"] = id_to_captions.get(int(img["id"]), [])
                    return entry, None

            # 3) If skipping downloads, report missing
            if skip_download:
                return None, {"file_name": file_name, "reason": "missing_local_file"}

            # 4) Try Open Images CLI if permitted
            if openimg_id and use_openimages_cli:
                nonlocal _openimages_ready
                if _openimages_ready is None:
                    _openimages_ready = _ensure_openimages_cli(pip_install_openimages)
                if _openimages_ready:
                    tmp_dir = image_root / "_tmp_openimages" / split
                    downloaded = _download_openimages_id(str(openimg_id), tmp_dir, timeout=timeout)
                    if downloaded and _try_copy(downloaded, dest):
                        entry = {"image": f"{split}/{file_name}", "img_id": int(img["id"])}
                        if id_to_captions:
                            entry["captions"] = id_to_captions.get(int(img["id"]), [])
                        return entry, None

            # 5) Fallback to HTTP if URL exists
            if url:
                success, err = fetch_image(url, dest, retries=retries, timeout=timeout)
                if not success:
                    return None, {"file_name": file_name, "reason": err}
            else:
                return None, {"file_name": file_name, "reason": "no_source"}

        entry = {"image": f"{split}/{file_name}", "img_id": int(img["id"])}
        if id_to_captions:
            entry["captions"] = id_to_captions.get(int(img["id"]), [])
        return entry, None

    progress = tqdm(total=len(images), desc=f"{split} images", unit="img")

    def handle_result(result: Tuple[Optional[Dict], Optional[Dict]]) -> None:
        entry, failure = result
        if entry:
            converted.append(entry)
        if failure:
            failures.append(failure)
        progress.update(1)

    worker_count = max(1, workers)
    if worker_count == 1:
        for img in images:
            handle_result(process_image(img))
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(process_image, img) for img in images]
            for future in as_completed(futures):
                handle_result(future.result())

    progress.close()
    converted.sort(key=lambda x: x["img_id"])

    return converted, failures


def write_config(config_template: Path, image_root: Path, ann_root: Path, batch_size: int, config_out: Path) -> Path:
    config_template = config_template.resolve()
    with open(config_template, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    config["image_root"] = str(image_root.resolve()).replace("\\", "/")
    config["ann_root"] = str(ann_root.resolve()).replace("\\", "/")
    config["batch_size"] = batch_size

    config_out = config_out.resolve()
    config_out.parent.mkdir(parents=True, exist_ok=True)
    with open(config_out, "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    return config_out


def run_eval(config_path: Path, output_dir: Path, device: str, seed: int) -> None:
    """Invoke eval_nocaps using the freshly written config."""
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    result_dir = output_dir / "result"

    output_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    eval_args = SimpleNamespace(
        config=str(config_path),
        output_dir=str(output_dir),
        result_dir=str(result_dir),
        device=device,
        seed=seed,
        world_size=1,
        dist_url="env://",
        distributed=False,
        gpu=0,
    )

    # Mirror the CLI script behaviour for reference dumps.
    with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    eval_nocaps.main(eval_args, config)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare NoCaps data and run BLIP evaluation locally.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Directory containing nocap_val_4500_captions.json and nocaps_test_image_info.json.",
    )
    parser.add_argument("--val-json", default="nocaps_val_4500_captions.json", help="Validation JSON filename.")
    parser.add_argument("--test-json", default="nocaps_test_image_info.json", help="Test JSON filename.")
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("data/nocaps_images"),
        help="Folder where images will be stored (downloaded if missing).",
    )
    parser.add_argument(
        "--ann-root",
        type=Path,
        default=Path("data/nocaps_annotations"),
        help="Folder where BLIP-ready annotation JSON files will be written.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/NoCaps"),
        help="Directory to store evaluation outputs (matches eval_nocaps.py default).",
    )
    parser.add_argument(
        "--config-template",
        type=Path,
        default=Path("configs/nocaps.yaml"),
        help="Base YAML config that will be adapted with local paths.",
    )
    parser.add_argument(
        "--config-out",
        type=Path,
        default=None,
        help="Optional path for the generated config; defaults to <output-dir>/nocaps_eval.yaml.",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size override for evaluation.")
    parser.add_argument("--retries", type=int, default=3, help="Number of download retries per image.")
    parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout (seconds) per image request.")
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent download workers (1 disables parallelism).",
    )
    parser.add_argument("--device", default="cuda", help="Device string passed to eval_nocaps (e.g. cuda or cpu).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed forwarded to eval_nocaps.")
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Only prepare data/config without running eval_nocaps.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip HTTP downloads and assume images are already present under --image-root.",
    )
    # Optional local dataset roots for mapping instead of downloading
    parser.add_argument(
        "--coco-train-root",
        type=Path,
        default=None,
        help="Path to COCO train2014 images to avoid re-downloading.",
    )
    parser.add_argument(
        "--coco-val-root",
        type=Path,
        default=None,
        help="Path to COCO val2014 images to avoid re-downloading.",
    )
    parser.add_argument(
        "--openimages-root",
        type=Path,
        default=None,
        help="Path to a local Open Images dataset for ID-based mapping.",
    )
    parser.add_argument(
        "--use-openimages-cli",
        action="store_true",
        help="Use the 'openimages' CLI to download images by ID when missing.",
    )
    parser.add_argument(
        "--pip-install-openimages",
        action="store_true",
        help="Attempt to pip install the 'openimages' package if CLI is unavailable.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    dataset_root = args.dataset_root.resolve()
    val_json = dataset_root / args.val_json
    test_json = dataset_root / args.test_json

    if not val_json.exists() or not test_json.exists():
        raise FileNotFoundError(
            f"Could not find required JSON files under {dataset_root}. "
            "Make sure nocap_val_4500_captions.json and nocaps_test_image_info.json are available."
        )

    image_root = args.image_root.resolve()
    ann_root = args.ann_root.resolve()
    output_dir = args.output_dir.resolve()

    (image_root / "val").mkdir(parents=True, exist_ok=True)
    (image_root / "test").mkdir(parents=True, exist_ok=True)
    ann_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(val_json, "r", encoding="utf-8") as f:
        val_raw = json.load(f)
    with open(test_json, "r", encoding="utf-8") as f:
        test_raw = json.load(f)

    # Build Open Images index if provided
    openimages_index = _index_openimages(args.openimages_root.resolve() if args.openimages_root else None)

    val_entries, val_failures = convert_split(
        val_raw,
        "val",
        image_root,
        retries=args.retries,
        timeout=args.timeout,
        skip_download=args.no_download,
        workers=args.workers,
        coco_train_root=(args.coco_train_root.resolve() if args.coco_train_root else None),
        coco_val_root=(args.coco_val_root.resolve() if args.coco_val_root else None),
        openimages_index=openimages_index,
        use_openimages_cli=bool(args.use_openimages_cli),
        pip_install_openimages=bool(args.pip_install_openimages),
    )
    test_entries, test_failures = convert_split(
        test_raw,
        "test",
        image_root,
        retries=args.retries,
        timeout=args.timeout,
        skip_download=args.no_download,
        workers=args.workers,
        coco_train_root=(args.coco_train_root.resolve() if args.coco_train_root else None),
        coco_val_root=(args.coco_val_root.resolve() if args.coco_val_root else None),
        openimages_index=openimages_index,
        use_openimages_cli=bool(args.use_openimages_cli),
        pip_install_openimages=bool(args.pip_install_openimages),
    )

    val_out = ann_root / "nocaps_val.json"
    test_out = ann_root / "nocaps_test.json"

    with open(val_out, "w", encoding="utf-8") as f:
        json.dump(val_entries, f)
    with open(test_out, "w", encoding="utf-8") as f:
        json.dump(test_entries, f)

    failures_log = output_dir / "download_failures.json"
    with open(failures_log, "w", encoding="utf-8") as f:
        json.dump({"val": val_failures, "test": test_failures}, f, indent=2)

    print(f"Validation annotations: {len(val_entries)} -> {val_out}")
    print(f"Test annotations: {len(test_entries)} -> {test_out}")
    print(f"Download failures: val={len(val_failures)}, test={len(test_failures)} (see {failures_log})")

    config_out = args.config_out or (output_dir / "nocaps_eval.yaml")
    config_path = write_config(args.config_template, image_root, ann_root, args.batch_size, config_out)
    print(f"Config saved to: {config_path}")

    if args.skip_eval:
        print("Skipping eval_nocaps (per --skip-eval).")
        return

    run_eval(config_path, output_dir, args.device, args.seed)
    print(f"Evaluation finished. Results stored under: {output_dir / 'result'}")


if __name__ == "__main__":
    main(sys.argv[1:])
