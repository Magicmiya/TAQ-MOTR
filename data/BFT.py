import os

from .dancetrack import DanceTrack


class BFT(DanceTrack):

    def __init__(self, config: dict, mode: str, transform=None):
        dataset_cfg = dict(config)
        dataset_cfg.setdefault("dataset_dir", os.path.join(dataset_cfg["dataset_root"], "BFT"))
        dataset_cfg.setdefault("video_name_prefix", "")
        dataset_cfg.setdefault("image_name_width", 8)
        # Preprocessed BFT follows the DanceTrack img1/gt layout and reuses its complete sampling pipeline.
        super(BFT, self).__init__(config=dataset_cfg, mode=mode, transform=transform)

    def _get_eval_frame_ids(self, vid: str) -> list[int]:
        """Evaluate every image frame, including BFT test frames without GT objects."""
        img_dir = os.path.join(self.split_dir, vid, "img1")
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Image directory not found for video {vid}: {img_dir}")

        frame_ids = []
        for file_name in sorted(os.listdir(img_dir)):
            stem, ext = os.path.splitext(file_name)
            if ext.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            if stem.isdigit():
                frame_ids.append(int(stem))

        if not frame_ids:
            raise FileNotFoundError(f"No image frames found for video {vid}: {img_dir}")
        return frame_ids


def build(config: dict, mode: str):
    return BFT(config=config, mode=mode)
