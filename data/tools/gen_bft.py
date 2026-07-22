import argparse
import os
from pathlib import Path

from PIL import Image


SPLITS = ("train", "val", "test")


def parse_args():
    parser = argparse.ArgumentParser(
        "Convert BFT v1.5 to the DanceTrack directory layout",
        add_help=True,
    )
    parser.add_argument(
        "-P",
        "--data_path",
        type=Path,
        required=True,
        help="Dataset root containing the BFT directory.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
        help="Dataset splits to preprocess.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images instead of creating relative symbolic links.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace generated files when their content or link target differs.",
    )
    return parser.parse_args()


def _read_mot_rows(annotation_path: Path) -> list[str]:
    rows = []
    with annotation_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            fields = [field.strip() for field in line.strip().split(",")]
            if not line.strip():
                continue
            if len(fields) < 6:
                raise ValueError(
                    f"{annotation_path}:{line_number} has {len(fields)} fields; expected at least 6."
                )
            frame_id, track_id = int(fields[0]), int(fields[1])
            if frame_id < 1 or track_id < 0:
                raise ValueError(
                    f"{annotation_path}:{line_number} has invalid frame/id: {frame_id}, {track_id}."
                )
            x, y, width, height = (float(value) for value in fields[2:6])
            if width <= 0 or height <= 0:
                raise ValueError(
                    f"{annotation_path}:{line_number} has a non-positive box: {fields[2:6]}."
                )
            # BFT uses -1 placeholders; DanceTrack/TrackEval require marked-valid, single-class GT.
            rows.append(
                f"{frame_id},{track_id},{_format_number(x)},{_format_number(y)},"
                f"{_format_number(width)},{_format_number(height)},1,1,1"
            )
    return rows


def _format_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _write_text(path: Path, content: str, force: bool):
    if path.exists():
        if path.is_file() and path.read_text(encoding="utf-8") == content:
            return "unchanged"
        if not force:
            raise FileExistsError(f"{path} exists with different content; pass --force to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return "write"


def _stage_image(source: Path, target: Path, copy_images: bool, force: bool):
    if target.is_symlink():
        expected = os.path.relpath(source, start=target.parent)
        if os.readlink(target) == expected:
            return "unchanged"
        if not force:
            raise FileExistsError(f"{target} points elsewhere; pass --force to replace it.")
        target.unlink()
    elif target.exists():
        if copy_images and target.is_file() and target.stat().st_size == source.stat().st_size:
            return "unchanged"
        if not force:
            raise FileExistsError(f"{target} already exists; pass --force to replace it.")
        target.unlink()

    target.parent.mkdir(parents=True, exist_ok=True)
    if copy_images:
        import shutil

        shutil.copy2(source, target)
        return "copy"

    target.symlink_to(os.path.relpath(source, start=target.parent))
    return "link"


def _build_seqinfo(sequence_name: str, frame_count: int, width: int, height: int) -> str:
    return (
        "[Sequence]\n"
        f"name={sequence_name}\n"
        "imDir=img1\n"
        "frameRate=30\n"
        f"seqLength={frame_count}\n"
        f"imWidth={width}\n"
        f"imHeight={height}\n"
        "imExt=.jpg\n"
    )


def preprocess_sequence(
    sequence_dir: Path,
    annotation_path: Path,
    copy_images: bool,
    force: bool,
) -> dict[str, int]:
    source_images = sorted(sequence_dir.glob("*.jpg"))
    if not source_images:
        raise FileNotFoundError(f"No JPG frames found in {sequence_dir}.")

    frame_ids = [int(path.stem) for path in source_images if path.stem.isdigit()]
    expected_ids = list(range(1, len(source_images) + 1))
    if len(frame_ids) != len(source_images) or frame_ids != expected_ids:
        raise ValueError(f"{sequence_dir} must contain contiguous numeric JPG frames starting at 1.")
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    with Image.open(source_images[0]) as image:
        width, height = image.size

    actions: dict[str, int] = {}
    for frame_id, source in zip(frame_ids, source_images):
        target = sequence_dir / "img1" / f"{frame_id:08d}.jpg"
        action = _stage_image(
            source=source,
            target=target,
            copy_images=copy_images,
            force=force,
        )
        actions[action] = actions.get(action, 0) + 1

    gt_content = "\n".join(_read_mot_rows(annotation_path)) + "\n"
    gt_action = _write_text(
        sequence_dir / "gt" / "gt.txt",
        gt_content,
        force=force,
    )
    actions[f"gt_{gt_action}"] = actions.get(f"gt_{gt_action}", 0) + 1

    seqinfo_content = _build_seqinfo(
        sequence_name=sequence_dir.name,
        frame_count=len(source_images),
        width=width,
        height=height,
    )
    seqinfo_action = _write_text(
        sequence_dir / "seqinfo.ini",
        seqinfo_content,
        force=force,
    )
    actions[f"seqinfo_{seqinfo_action}"] = actions.get(f"seqinfo_{seqinfo_action}", 0) + 1
    return actions


def gen_bft(
    data_root: Path,
    splits: list[str],
    copy_images: bool = False,
    force: bool = False,
):
    dataset_root = data_root.expanduser().resolve() / "BFT"
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"BFT dataset not found: {dataset_root}")

    total_sequences = 0
    total_actions: dict[str, int] = {}
    for split in splits:
        split_dir = dataset_root / split
        annotations_dir = dataset_root / "annotations_mot" / split
        if not split_dir.is_dir() or not annotations_dir.is_dir():
            raise FileNotFoundError(f"Incomplete BFT split: {split}")

        sequences = sorted(path for path in split_dir.iterdir() if path.is_dir())
        for sequence_dir in sequences:
            actions = preprocess_sequence(
                sequence_dir=sequence_dir,
                annotation_path=annotations_dir / f"{sequence_dir.name}.txt",
                copy_images=copy_images,
                force=force,
            )
            total_sequences += 1
            for action, count in actions.items():
                total_actions[action] = total_actions.get(action, 0) + count
        print(f"{split}: validated {len(sequences)} sequences")

    rendered_actions = ", ".join(f"{key}={value}" for key, value in sorted(total_actions.items()))
    print(f"Preprocessing finished: sequences={total_sequences}, {rendered_actions}")


if __name__ == "__main__":
    options = parse_args()
    gen_bft(
        data_root=options.data_path,
        splits=options.splits,
        copy_images=options.copy_images,
        force=options.force,
    )
