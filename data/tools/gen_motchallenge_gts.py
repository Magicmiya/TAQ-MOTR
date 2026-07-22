import argparse
import configparser
import json
import shutil
from pathlib import Path

import numpy as np


JSON_SPLITS = ("train", "train_half", "val_half", "test")
CATEGORIES = [
    {"id": 1, "name": "pedestrian"},
    {"id": 2, "name": "person on vehicle"},
    {"id": 3, "name": "car"},
    {"id": 4, "name": "bicycle"},
    {"id": 5, "name": "motorbike"},
    {"id": 6, "name": "non motorized vehicle"},
    {"id": 7, "name": "static person"},
    {"id": 8, "name": "distractor"},
    {"id": 9, "name": "occluder"},
    {"id": 10, "name": "occluder on the ground"},
    {"id": 11, "name": "occluder full"},
    {"id": 12, "name": "reflection"},
]
BENCHMARK_SPECS = {
    "MOT17": {
        "detectors": ("DPM", "FRCNN", "SDP"),
        "primary_detector": "FRCNN",
        "train_half_gt": "gt_train_half.txt",
        "val_half_gt": "gt_val_half.txt",
        "train_half_det": "det_train_half.txt",
        "val_half_det": "det_val_half.txt",
    },
    "MOT20": {
        "detectors": (),
        "primary_detector": None,
        "train_half_gt": "gt_half-train.txt",
        "val_half_gt": "gt_half-val.txt",
        "train_half_det": "det_half-train.txt",
        "val_half_det": "det_half-val.txt",
    },
}


def parse_option():
    parser = argparse.ArgumentParser("Generate MOTChallenge half splits and COCO annotations", add_help=True)
    parser.add_argument("-P", "--data_path", type=str, help="datasets root path", required=True)
    parser.add_argument("-B", "--benchmark", choices=tuple(BENCHMARK_SPECS.keys()), required=True)
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.loadtxt(path, dtype=np.float64, delimiter=",", ndmin=2)


def read_seqinfo(video_dir: Path) -> dict[str, str]:
    parser = configparser.ConfigParser()
    parser.read(video_dir / "seqinfo.ini")
    return {key.lower(): value for key, value in parser["Sequence"].items()}


def get_seqinfo_value(video_info: dict[str, str], key: str, default=None):
    return video_info.get(key.lower(), default)


def get_image_count(video_dir: Path) -> int:
    img_dir = video_dir / "img1"
    return sum(1 for file_path in img_dir.iterdir() if file_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})


def get_half_range(num_images: int, split_name: str) -> tuple[int, int]:
    if split_name == "train_half":
        return 0, num_images // 2 - 1
    if split_name == "val_half":
        return num_images // 2, num_images - 1
    raise ValueError(f"Unsupported half split: {split_name}")


def write_txt(path: Path, rows: np.ndarray):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            values = row.tolist()
            formatted = [str(int(values[0])), str(int(values[1]))]
            for value in values[2:]:
                float_value = float(value)
                if abs(float_value - round(float_value)) < 1e-9:
                    formatted.append(str(int(round(float_value))))
                else:
                    formatted.append(f"{float_value:.6f}".rstrip("0").rstrip("."))
            f.write(",".join(formatted) + "\n")


def list_primary_videos(dataset_root: Path, benchmark: str) -> list[Path]:
    train_root = dataset_root / "train"
    spec = BENCHMARK_SPECS[benchmark]
    detectors = spec["detectors"]
    if not detectors:
        return sorted([path for path in train_root.iterdir() if path.is_dir() and path.name.startswith(benchmark)])
    suffix = f"-{spec['primary_detector']}"
    return sorted([path for path in train_root.iterdir() if path.is_dir() and path.name.endswith(suffix)])


def list_split_videos(dataset_root: Path, benchmark: str, split_name: str) -> list[Path]:
    split_root = dataset_root / ("test" if split_name == "test" else "train")
    spec = BENCHMARK_SPECS[benchmark]
    detectors = spec["detectors"]
    if split_name == "test":
        if not detectors:
            return sorted([path for path in split_root.iterdir() if path.is_dir() and path.name.startswith(benchmark)])
        suffix = f"-{spec['primary_detector']}"
        return sorted([path for path in split_root.iterdir() if path.is_dir() and path.name.endswith(suffix)])
    return list_primary_videos(dataset_root=dataset_root, benchmark=benchmark)


def write_half_split_files(dataset_root: Path, benchmark: str):
    train_root = dataset_root / "train"
    spec = BENCHMARK_SPECS[benchmark]
    for primary_video in list_primary_videos(dataset_root=dataset_root, benchmark=benchmark):
        video_info = read_seqinfo(primary_video)
        num_images = int(get_seqinfo_value(video_info, "seqLength"))
        gt_full = load_csv(primary_video / "gt" / "gt.txt")

        for split_name, gt_file_name, det_file_name in (
            ("train_half", spec["train_half_gt"], spec["train_half_det"]),
            ("val_half", spec["val_half_gt"], spec["val_half_det"]),
        ):
            start, end = get_half_range(num_images=num_images, split_name=split_name)
            gt_half = np.asarray(
                [row.copy() for row in gt_full if start <= int(row[0] - 1) <= end],
                dtype=np.float64,
            )
            if gt_half.size > 0:
                gt_half[:, 0] -= start

            gt_target = primary_video / "gt" / gt_file_name
            write_txt(gt_target, gt_half)

            detectors = spec["detectors"]
            if detectors:
                base_name = primary_video.name.rsplit("-", 1)[0]
                target_videos = [train_root / f"{base_name}-{detector}" for detector in detectors]
            else:
                target_videos = [primary_video]

            for target_video in target_videos:
                if not target_video.is_dir():
                    continue
                if target_video != primary_video:
                    shutil.copy2(gt_target, target_video / "gt" / gt_file_name)

                det_path = target_video / "det" / "det.txt"
                if not det_path.is_file():
                    continue
                det_full = load_csv(det_path)
                det_half = np.asarray(
                    [row.copy() for row in det_full if start <= int(row[0] - 1) <= end],
                    dtype=np.float64,
                )
                if det_half.size > 0:
                    det_half[:, 0] -= start
                write_txt(target_video / "det" / det_file_name, det_half)


def build_json_split(dataset_root: Path, benchmark: str, split_name: str):
    annotations_root = dataset_root / "annotations"
    ensure_dir(annotations_root)
    videos = list_split_videos(dataset_root=dataset_root, benchmark=benchmark, split_name=split_name)

    out = {"images": [], "annotations": [], "videos": [], "categories": CATEGORIES}
    image_cnt = 0
    ann_cnt = 0
    video_cnt = 0
    track_cnt = 0

    for video_dir in videos:
        video_cnt += 1
        video_info = read_seqinfo(video_dir)
        num_images = get_image_count(video_dir)
        if split_name in {"train_half", "val_half"}:
            image_range = get_half_range(num_images=num_images, split_name=split_name)
        else:
            image_range = (0, num_images - 1)

        seq_length = image_range[1] - image_range[0] + 1
        image_ext = get_seqinfo_value(video_info, "imExt", ".jpg")
        out["videos"].append(
            {
                "id": video_cnt,
                "file_name": video_dir.name,
                "frameRate": int(get_seqinfo_value(video_info, "frameRate", 0)),
                "imWidth": int(get_seqinfo_value(video_info, "imWidth", 0)),
                "imHeight": int(get_seqinfo_value(video_info, "imHeight", 0)),
                "seqLength": seq_length,
            }
        )

        frame_to_image_id = {}
        for local_idx, frame_id in enumerate(range(image_range[0] + 1, image_range[1] + 2), start=1):
            image_cnt += 1
            frame_to_image_id[local_idx] = image_cnt
            out["images"].append(
                {
                    "file_name": f"{video_dir.name}/img1/{frame_id:06d}{image_ext}",
                    "id": image_cnt,
                    "frame_id": local_idx,
                    "prev_image_id": image_cnt - 1 if local_idx > 1 else -1,
                    "next_image_id": image_cnt + 1 if local_idx < seq_length else -1,
                    "video_id": video_cnt,
                    "height": int(get_seqinfo_value(video_info, "imHeight", 0)),
                    "width": int(get_seqinfo_value(video_info, "imWidth", 0)),
                }
            )

        if split_name == "test":
            continue

        gt_full = load_csv(video_dir / "gt" / "gt.txt")
        track_id_map: dict[int, int] = {}
        for row in gt_full:
            frame_id = int(row[0])
            if not (image_range[0] <= frame_id - 1 <= image_range[1]):
                continue
            local_frame_id = frame_id - image_range[0]
            original_track_id = int(row[1])
            if original_track_id not in track_id_map:
                track_cnt += 1
                track_id_map[original_track_id] = track_cnt

            ann_cnt += 1
            out["annotations"].append(
                {
                    "id": ann_cnt,
                    "category_id": int(row[7]),
                    "image_id": frame_to_image_id[local_frame_id],
                    "track_id": track_id_map[original_track_id],
                    "bbox": row[2:6].tolist(),
                    "conf": float(row[8]),
                    "iscrowd": 0 if int(row[6]) == 1 else 1,
                    "area": float(row[4] * row[5]),
                }
            )

    out_path = annotations_root / f"{split_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f)


def process_benchmark(datasets_root: str | Path, benchmark: str):
    datasets_root = Path(datasets_root).expanduser().resolve()
    if not datasets_root.is_dir():
        raise FileNotFoundError(f"Datasets root not found: {datasets_root}")
    dataset_root = datasets_root / benchmark
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"{benchmark} dataset not found under: {datasets_root}")

    # Keep MOT17 and MOT20 preprocessing in one implementation so split generation rules stay aligned.
    write_half_split_files(dataset_root=dataset_root, benchmark=benchmark)
    for split_name in JSON_SPLITS:
        build_json_split(dataset_root=dataset_root, benchmark=benchmark, split_name=split_name)
    print(f"{benchmark} preprocessing finished: {dataset_root}")


def main():
    args = parse_option()
    process_benchmark(datasets_root=args.data_path, benchmark=args.benchmark)


if __name__ == "__main__":
    main()
