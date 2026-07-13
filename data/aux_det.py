import json
import os
from collections import defaultdict


class AuxDetSource:
    def __init__(self, name: str, items: list[dict]):
        self.name = str(name)
        self.items = list(items)

    def __len__(self) -> int:
        return len(self.items)

    def get_item(self, index: int) -> dict:
        if not self.items:
            raise IndexError(f"Aux detection source '{self.name}' is empty.")
        return self.items[int(index) % len(self.items)]


def _resolve_image_path(file_name: str, dataset_root: str) -> str:
    if os.path.isabs(file_name):
        return file_name
    return os.path.join(dataset_root, file_name)


def load_coco_aux_det_source(
    name: str,
    dataset_root: str,
    annotations_path: str,
) -> AuxDetSource:
    if not os.path.isfile(annotations_path):
        raise FileNotFoundError(
            f"Missing aux detection annotations for {name}: {annotations_path}"
        )

    with open(annotations_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = {}
    for image in data.get("images", []):
        image_id = int(image["id"])
        images[image_id] = {
            "img_path": _resolve_image_path(str(image["file_name"]), dataset_root),
            "width": int(image["width"]),
            "height": int(image["height"]),
        }

    ann_map: dict[int, list[dict]] = defaultdict(list)
    for ann in data.get("annotations", []):
        if int(ann.get("category_id", 1)) != 1:
            continue
        if int(ann.get("iscrowd", 0)) != 0:
            continue
        image_id = int(ann["image_id"])
        bbox = ann.get("bbox", None)
        if bbox is None or len(bbox) != 4:
            continue
        x, y, w, h = map(float, bbox)
        if w <= 0 or h <= 0:
            continue
        ann_map[image_id].append({"bbox": [x, y, w, h]})

    items = []
    for image_id, image in images.items():
        img_path = image["img_path"]
        if not os.path.isfile(img_path):
            raise FileNotFoundError(
                f"Aux detection image not found for {name}: {img_path}"
            )
        items.append(
            {
                "image_id": image_id,
                "img_path": img_path,
                "width": image["width"],
                "height": image["height"],
                "annotations": ann_map.get(image_id, []),
            }
        )

    if not items:
        raise RuntimeError(f"Aux detection source '{name}' has no valid items in {annotations_path}.")

    return AuxDetSource(name=name, items=items)
