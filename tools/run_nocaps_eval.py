import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import requests
try:
    import ruamel_yaml as yaml
except ModuleNotFoundError:  # pragma: no cover - optional dependency name
    from ruamel import yaml  # type: ignore
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


def convert_split(
    raw_data: Dict,
    split: str,
    image_root: Path,
    retries: int,
    timeout: int,
    skip_download: bool,
    workers: int,
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

    def process_image(img: Dict) -> Tuple[Optional[Dict], Optional[Dict]]:
        file_name = img["file_name"]
        url = img.get("coco_url") or img.get("flickr_url")
        if not url:
            return None, {"file_name": file_name, "reason": "missing_url"}

        dest = image_root / split / file_name
        if not dest.exists():
            if skip_download:
                return None, {"file_name": file_name, "reason": "missing_local_file"}
            success, err = fetch_image(url, dest, retries=retries, timeout=timeout)
            if not success:
                return None, {"file_name": file_name, "reason": err}

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

    val_entries, val_failures = convert_split(
        val_raw,
        "val",
        image_root,
        retries=args.retries,
        timeout=args.timeout,
        skip_download=args.no_download,
        workers=args.workers,
    )
    test_entries, test_failures = convert_split(
        test_raw,
        "test",
        image_root,
        retries=args.retries,
        timeout=args.timeout,
        skip_download=args.no_download,
        workers=args.workers,
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
