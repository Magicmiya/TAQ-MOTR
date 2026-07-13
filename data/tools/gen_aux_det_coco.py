import argparse
import json
import os
from pathlib import Path

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser("Generate COCO-style annotations for aux detection datasets", add_help=True)
    parser.add_argument("-P", "--data_path", type=str, required=True, help="Datasets root path")
    parser.add_argument(
        "-D",
        "--dataset",
        type=str,
        choices=["CityPersons", "CrowdHuman"],
        required=True,
        help="Aux detection dataset name",
    )
    parser.add_argument("-O", "--output_path", type=str, default=None, help="Optional output directory override")
    return parser.parse_args()


def load_odgt_records(annotation_path: Path) -> list[dict]:
    with annotation_path.open("r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def build_citypersons(datasets_root: Path, output_dir: Path | None = None):
    dataset_root = datasets_root / "CityPersons"
    data_list_path = dataset_root / "citypersons.train"
    out_dir = dataset_root / "annotations" if output_dir is None else output_dir
    out_path = out_dir / "train.json"
    if not data_list_path.is_file():
        raise FileNotFoundError(f"CityPersons train list not found: {data_list_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with data_list_path.open("r", encoding="utf-8") as f:
        image_paths = [line.strip() for line in f if line.strip()]

    images = []
    annotations = []
    image_count = 0
    ann_count = 0
    for image_rel_path in image_paths:
        image_count += 1
        normalized_rel_path = image_rel_path.replace("Cityscapes/", "", 1)
        img_path = dataset_root / normalized_rel_path
        label_path = dataset_root / normalized_rel_path.replace("images/", "labels_with_ids/")
        label_path = label_path.with_suffix(".txt")
        if not img_path.is_file():
            raise FileNotFoundError(f"CityPersons image not found: {img_path}")

        width, height = Image.open(img_path).size
        images.append(
            {
                "file_name": str(img_path.resolve()),
                "id": image_count,
                "height": int(height),
                "width": int(width),
            }
        )

        if not label_path.is_file():
            continue
        with label_path.open("r", encoding="utf-8") as f:
            for line in f:
                fields = line.strip().split()
                if len(fields) != 6:
                    continue
                _, _, cx, cy, bw, bh = map(float, fields)
                x = width * (cx - bw / 2.0)
                y = height * (cy - bh / 2.0)
                w = width * bw
                h = height * bh
                if w <= 0 or h <= 0:
                    continue
                ann_count += 1
                annotations.append(
                    {
                        "id": ann_count,
                        "category_id": 1,
                        "image_id": image_count,
                        "track_id": -1,
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                    }
                )

    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "person"}]}, f)
    print(f"CityPersons annotations saved to {out_path}")


def resolve_crowdhuman_image(dataset_root: Path, split: str, image_id: str) -> Path:
    split_dirs = sorted(dataset_root.glob(f"CrowdHuman_{split}*/Images"))
    for image_dir in split_dirs:
        img_path = image_dir / f"{image_id}.jpg"
        if img_path.is_file():
            return img_path
    raise FileNotFoundError(f"CrowdHuman image not found for split={split}, image_id={image_id}")


def build_crowdhuman(datasets_root: Path, output_dir: Path | None = None):
    dataset_root = datasets_root / "CrowdHuman"
    out_dir = dataset_root / "annotations" if output_dir is None else output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        annotation_path = dataset_root / f"annotation_{split}.odgt"
        if not annotation_path.is_file():
            raise FileNotFoundError(f"CrowdHuman annotation file not found: {annotation_path}")

        records = load_odgt_records(annotation_path)
        images = []
        annotations = []
        image_count = 0
        ann_count = 0
        for record in records:
            image_count += 1
            image_id = str(record["ID"])
            img_path = resolve_crowdhuman_image(dataset_root=dataset_root, split=split, image_id=image_id)
            width, height = Image.open(img_path).size
            images.append(
                {
                    "file_name": str(img_path.resolve()),
                    "id": image_count,
                    "height": int(height),
                    "width": int(width),
                }
            )

            for ann in record.get("gtboxes", []):
                fbox = ann.get("fbox", None)
                if fbox is None or len(fbox) != 4:
                    continue
                is_ignore = int(ann.get("extra", {}).get("ignore", 0)) == 1
                x, y, w, h = map(float, fbox)
                if w <= 0 or h <= 0:
                    continue
                ann_count += 1
                annotations.append(
                    {
                        "id": ann_count,
                        "category_id": 1,
                        "image_id": image_count,
                        "track_id": -1,
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 1 if is_ignore else 0,
                    }
                )

        out_path = out_dir / f"{split}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump({"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "person"}]}, f)
        print(f"CrowdHuman {split} annotations saved to {out_path}")


def main():
    args = parse_args()
    datasets_root = Path(args.data_path).expanduser().resolve()
    if not datasets_root.is_dir():
        raise FileNotFoundError(f"Datasets root not found: {datasets_root}")
    output_dir = None if args.output_path is None else Path(args.output_path).expanduser().resolve()

    if args.dataset == "CityPersons":
        build_citypersons(datasets_root, output_dir=output_dir)
        return
    if args.dataset == "CrowdHuman":
        build_crowdhuman(datasets_root, output_dir=output_dir)
        return
    raise ValueError(f"Unsupported dataset: {args.dataset}")


if __name__ == "__main__":
    main()
