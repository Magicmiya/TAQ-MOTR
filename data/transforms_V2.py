import torch.nn
import collections.abc
import numpy as np
import math
# torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as v2
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import matplotlib.pyplot as plt

from torchvision.transforms.v2 import functional as F
from torchvision.ops.boxes import box_iou
from torchvision.utils import draw_bounding_boxes, draw_keypoints, draw_segmentation_masks
from torchvision.tv_tensors import Video, BoundingBoxFormat, BoundingBoxes
from torchvision.transforms.v2._utils import query_size

from .transfom_utils import get_muti_bounding_boxes

from typing import Any, Union, Optional, Sequence, Tuple, Dict, List, Iterable


class MultiComposeV2(v2.Compose):

    def __init__(self, transforms: Sequence, policy=None) -> None:
        super().__init__(transforms=transforms)
        if policy is None:
            policy = {'name': 'default', 'epoch': 1000000}
        self.policy = policy
        self.global_samples = 0

    def forward(self, first_input: Any, *rest_inputs: Any) -> Any:
        inputs = (first_input,) + rest_inputs if rest_inputs else first_input
        multi_resize_fc = next(
            (f for f in self.transforms if isinstance(f, RandomResize)), None)
        if multi_resize_fc is not None:
            dataset = inputs[-1]
            resize_idx = getattr(dataset, "random_idx", 0)
            multi_resize_fc.set_size_idx(resize_idx)

        inputs, info = self.convert_to_tv_tensor(*inputs)
        sample = self.get_forward(self.policy['name'])(*inputs)
        # show(sample)
        img = sample[0]
        h, w = img[0].shape[-2:]
        for k, v in sample[1].items():
            for i in range(len(info)):
                if k in info[i]:
                    info[i][k] = v[i]
                if k == 'bbox' and 'areas' in info[i]:
                    info[i]['areas'] = v[i][:, 2] * v[i][:, 3] * h * w
                pass
        return img, info

    def get_forward(self, name):
        forwards = {
            'default': self.default_forward,
            'stop_epoch': self.stop_epoch_forward,
            'stop_sample': self.stop_sample_forward,
        }
        return forwards[name]

    def default_forward(self, first_input: Any, *rest_inputs: Any) -> Any:
        sample = (first_input,) + rest_inputs if rest_inputs else first_input
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def stop_epoch_forward(self, first_input: Any, *rest_inputs: Any) -> Any:
        sample = (first_input,) + rest_inputs if rest_inputs else first_input
        dataset = sample[-1]

        cur_epoch = dataset.epoch
        policy_ops = self.policy['ops']
        policy_epoch = self.policy['epoch']
        skip_policy_ops = cur_epoch >= policy_epoch

        for transform in self.transforms:
            if type(transform
                    ).__name__ in policy_ops and skip_policy_ops:
                pass
            else:
                sample = transform(sample)
        return sample

    def stop_sample_forward(self, first_input: Any, *rest_inputs: Any):
        sample = (first_input,) + rest_inputs if rest_inputs else first_input
        dataset = sample[-1]

        cur_epoch = dataset.epoch
        policy_ops = self.policy['ops']
        policy_sample = self.policy['sample']

        for transform in self.transforms:
            if type(
                    transform
            ).__name__ in policy_ops and self.global_samples >= policy_sample:
                pass
            else:
                sample = transform(sample)

        self.global_samples += 1

        return sample

    @staticmethod
    def convert_to_tv_tensor(*inputs: Any):
        anno = {"bbox": [], "ids": [], "labels": [], "is_crowd": []}
        img, info = inputs[0], inputs[1]
        dataset = inputs[-1]
        for _, i in enumerate(info):
            w, h = i['org_shape']
            for k, v in i.items():
                if 'bbox' in k:
                    anno["bbox"].append(
                        BoundingBoxes(  # type: ignore
                            data=i['bbox'],
                            format=BoundingBoxFormat.XYXY,
                            canvas_size=(h, w)))
                    continue
                elif k in anno.keys():
                    anno[k].append(v)
                else:
                    pass
        anno = {key: value for key, value in anno.items() if len(value) > 0}
        # video = Video(torch.stack([F.to_image(i) for i in img], dim=0))
        return (img, anno, dataset), info


class DynamicRandomSelect(v2.Transform):

    def __init__(self, transform1: Sequence, transform2: Sequence, p: float = 0.5, rate_attr: str = "DE_random_IoU_crop_rate") -> None:
        super().__init__()
        self.transform1 = list(transform1)
        self.transform2 = list(transform2)
        self.p = float(p)
        self.rate_attr = rate_attr

    def forward(self, first_input: Any, *rest_inputs: Any) -> Any:
        sample = (first_input,) + rest_inputs if rest_inputs else first_input
        rate = self._get_rate(sample)
        transforms = self.transform1 if torch.rand(1).item() < rate else self.transform2
        for transform in transforms:
            sample = transform(sample)
        return sample

    def _get_rate(self, sample: Any) -> float:
        dataset = sample[-1] if isinstance(sample, (tuple, list)) and len(sample) > 0 else None
        rate = getattr(dataset, self.rate_attr, self.p)
        try:
            rate = float(rate)
        except (TypeError, ValueError):
            rate = self.p
        return max(0.0, min(1.0, rate))


class MultiSanitizeBoundingBoxes(v2.Transform):

    def __init__(self,
                 min_size: float = 1.0,
                 min_aspect: float = 0.1,
                 max_aspect: float = 10,
                 labels_key=None,
                 labels_getter=None):
        """
        The extension of sanitize_bounding_boxes is used to handle labels when the input is multi-frame images.
        At the same time, an aspect ratio condition item has been added to remove Bboxes and their labels that
        do not meet the requirements
        
        Args:
            min_size (float, optional): The size below which bounding boxes are removed. Default is 1.0.
            min_aspect (float, optional): Minimum aspect ratio (width/height) for bounding boxes. Boxes with aspect ratio 
                below this value will be removed. Default is 0.1.
            max_aspect (float, optional): Maximum aspect ratio (width/height) for bounding boxes. Boxes with aspect ratio 
                above this value will be removed. Default is 10.
            labels_key (str or list[str] or tuple[str], optional): Keys to identify labels in the input dictionary. 
                If None, defaults to ["bbox", "ids", "labels", "is_crowd"]. If str, will be converted to a list.
            labels_getter (callable or None, optional): Function to extract labels from inputs. If None, uses the default 
                _get_batch_labels method which expects a dictionary or two-tuple with dictionary as second element.
        """
        super().__init__()
        self.min_size = min_size
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.labels_key: list[str]
        if isinstance(labels_key, str):
            self.labels_key = [labels_key]
        elif isinstance(labels_key, (list, tuple)):
            for k in labels_key:
                assert isinstance(
                    k, str), "labels key must be str or list(tuple) of str"
            self.labels_key = list(labels_key)
        else:
            self.labels_key = ["bbox", "ids", "labels", "is_crowd"]
        assert 'bbox' in self.labels_key[0]
        if labels_getter is None:
            self.labels_getter = self._get_batch_labels

    @staticmethod
    def _get_batch_labels(inputs: Iterable[object], label_keys: list):
        if isinstance(inputs, (tuple, list)):
            _inputs = inputs[1]
        if not isinstance(_inputs, collections.abc.Mapping):
            raise ValueError(
                f"When using the default labels_getter, the input passed to forward must be a dictionary or a two-tuple "
                f"whose second item is a dictionary or a tensor, but got {_inputs} instead."
            )
        keys, labels = [], []
        for k, v in _inputs.items():
            if k in label_keys:
                labels.append(v)
                keys.append(k)
        return labels, keys

    def _get_sanitize_bounding_boxes_mask(self, bbox: BoundingBoxes,
                                          canvas_size: Tuple[int, int]):
        ws, hs = bbox[:, 2] - bbox[:, 0], bbox[:, 3] - bbox[:, 1]
        aspect = ws / hs
        valid = (ws >= self.min_size) & (hs >= self.min_size) & (
                aspect >= self.min_aspect) & (aspect <= self.max_aspect)
        image_h, image_w = canvas_size
        valid &= (bbox[:, 0] <= image_w) & (bbox[:, 2] >= 0)
        valid &= (bbox[:, 1] <= image_h) & (bbox[:, 3] >= 0)
        return valid

    def forward(self, first_input: Any, *rest_inputs: Any) -> Any:
        inputs = (first_input,) + rest_inputs if rest_inputs else first_input
        labels, keys = self.labels_getter(inputs, self.labels_key)
        labels_num = len(labels)
        seq_lens = len(labels[0])
        boxes_format = labels[0][0].format
        boxes_canvas = labels[0][0].canvas_size
        batch_labels_num = [box.shape[0] for box in labels[0]]
        labels = [torch.concat(_, dim=0) for _ in labels]
        boxes = BoundingBoxes(data=labels[0],  # type: ignore
                              format=boxes_format,
                              canvas_size=boxes_canvas)

        valid = self._get_sanitize_bounding_boxes_mask(boxes, boxes_canvas)

        labels = [label[valid] for label in labels]
        batch_labels_num = [
            torch.sum(valid[sum(batch_labels_num[:i]
                                ):sum(batch_labels_num[:i + 1])]).item()
            for i in range(seq_lens)
        ]

        _labels = [[] for _ in range(labels_num)]
        for i in range(seq_lens):
            start = sum(batch_labels_num[:i])
            end = sum(batch_labels_num[:i + 1])
            for j in range(labels_num):
                if j == 0:
                    boxes = BoundingBoxes(data=labels[j][start:end],  # type: ignore
                                          format=boxes_format,
                                          canvas_size=boxes_canvas)
                    _labels[j].append(boxes)
                else:
                    _labels[j].append(labels[j][start:end])
        for i, key in enumerate(keys):
            inputs[1][key] = _labels[i]
        return inputs


class MultiRandomIoUCrop(v2.RandomIoUCrop):

    def __init__(
            self,
            p: float = 0.5,
            min_scale: float = 0.3,
            max_scale: float = 1.0,
            min_aspect_ratio: float = 0.5,
            max_aspect_ratio: float = 2.0,
            sampler_options: Optional[List[float]] = None,
            trials: int = 40,
            visible_ratio: float = 0.5,
    ):
        """
        Extension of RandomIoUCrop for handling multi-frame image sequences.
        Adds a visible ratio constraint to ensure bounding boxes maintain sufficient visibility
        within the crop area across all frames.
        
        Args:
            p (float, optional): Probability of a sequence being randomly processed. Defaults to 0.5.
            min_scale (float, optional): Minimum scale factor for the crop size relative to the original image.
            max_scale (float, optional): Maximum scale factor for the crop size relative to the original image.
            min_aspect_ratio (float, optional): Minimum aspect ratio (width/height) for the crop.
            max_aspect_ratio (float, optional): Maximum aspect ratio (width/height) for the crop.
            sampler_options (Optional[List[float]], optional): List of IoU (Intersection over Union) threshold
                options for sampling valid crops. If None, uses default values.
            trials (int, optional): Maximum number of attempts to find a valid crop that meets all constraints.
            visible_ratio (float, optional): Minimum ratio of a bounding box that must be visible within the crop area.
                Boxes with visibility below this ratio will be filtered out. Default is 0.5.
        """
        super().__init__(min_scale, max_scale, min_aspect_ratio,
                         max_aspect_ratio, sampler_options, trials)
        self.p = p
        self.visible_ratio = visible_ratio

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        if torch.rand(1) >= self.p:
            return dict()
        orig_h, orig_w = query_size(flat_inputs)
        orig_aspect = orig_w / orig_h
        seq_bboxes = get_muti_bounding_boxes(flat_inputs)

        seq_bboxes_num = [bboxes.shape[0] for bboxes in seq_bboxes]
        seq_len = len(seq_bboxes_num)
        seq_bboxes = BoundingBoxes(  # type: ignore
            data=torch.concat(seq_bboxes, dim=0),  # type: ignore
            format=seq_bboxes[0].format,
            canvas_size=seq_bboxes[0].canvas_size,
        )
        # for no target sequence
        if seq_bboxes.shape[0] == 0:
            return dict()

        while True:
            # sample an option
            idx = int(torch.randint(low=0, high=len(self.options), size=(1,)))
            min_jaccard_overlap = self.options[idx]
            if min_jaccard_overlap >= 1.0:  # a value larger than 1 encodes the leave as-is option
                return dict()

            for _ in range(self.trials):
                # check the aspect ratio limitations
                r_h = self.min_scale + (self.max_scale -
                                        self.min_scale) * torch.rand(1)
                r_w_min = max(self.min_scale,
                              self.min_aspect_ratio * r_h / orig_aspect)
                r_w_max = min(self.max_scale,
                              self.max_aspect_ratio * r_h / orig_aspect)
                r_w = r_w_min + (r_w_max - r_w_min) * torch.rand(1)

                new_w = int(orig_w * r_w)
                new_h = int(orig_h * r_h)

                # check for 0 area crops
                r = torch.rand(2)
                left = int((orig_w - new_w) * r[0])
                top = int((orig_h - new_h) * r[1])
                right = left + new_w
                bottom = top + new_h
                if left == right or top == bottom:
                    continue

                # check for any valid boxes with centers within the crop area
                xyxy_bboxes = F.convert_bounding_box_format(
                    seq_bboxes.as_subclass(torch.Tensor),
                    seq_bboxes.format,
                    BoundingBoxFormat.XYXY,
                )
                w = (xyxy_bboxes[..., 2] - xyxy_bboxes[..., 0])
                h = (xyxy_bboxes[..., 3] - xyxy_bboxes[..., 1])

                visible_ratios = torch.stack(
                    [(xyxy_bboxes[..., 2] - left) / w,
                     (right - xyxy_bboxes[..., 0]) / w,
                     (xyxy_bboxes[..., 3] - top) / h,
                     (bottom - xyxy_bboxes[..., 1]) / h],
                    dim=-1)

                is_within_crop_area = (visible_ratios
                                       > self.visible_ratio).all(dim=-1)
                if not is_within_crop_area.any():
                    continue

                # check at least 1 box with jaccard limitations
                xyxy_bboxes = xyxy_bboxes[is_within_crop_area]
                ious = box_iou(
                    xyxy_bboxes,
                    torch.tensor([[left, top, right, bottom]],
                                 dtype=xyxy_bboxes.dtype,
                                 device=xyxy_bboxes.device),
                )
                if ious.max() < min_jaccard_overlap:
                    continue
                is_within_crop_areas = [
                    is_within_crop_area[sum(seq_bboxes_num[:n]
                                            ):sum(seq_bboxes_num[:n + 1])]
                    for n in range(seq_len)
                ]
                return dict(top=top,
                            left=left,
                            height=new_h,
                            width=new_w,
                            is_within_crop_area=is_within_crop_areas,
                            idx=0)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if len(params) < 1:
            return inpt

        output = self._call_kernel(F.crop,
                                   inpt,
                                   top=params["top"],
                                   left=params["left"],
                                   height=params["height"],
                                   width=params["width"])
        pass
        if isinstance(output, BoundingBoxes):
            # We "mark" the invalid boxes as degenreate, and they can be
            # removed by a later call to SanitizeBoundingBoxes()
            output[~params["is_within_crop_area"][params['idx']]] = 0
            params['idx'] += 1
        return output


class ConvertBoxes(v2.Transform):

    def __init__(self,
                 fmt: Optional[BoundingBoxFormat],
                 normalize=False) -> None:
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if not isinstance(inpt, BoundingBoxes):
            return inpt
        h, w = inpt.canvas_size
        old_format = inpt.format
        if self.normalize:
            inpt = inpt / torch.tensor([w, h, w, h])
        if self.fmt:
            inpt = F.convert_bounding_box_format(inpt=inpt,
                                                 old_format=old_format,
                                                 new_format=self.fmt)

        return inpt


class AlignFrameSequenceSize(v2.Transform):

    def __init__(self, size_divisibility: int | None = 32) -> None:
        super().__init__()
        self.size_divisibility = size_divisibility

    def forward(self, first_input: Any, *rest_inputs: Any) -> Any:
        """Pad frames in a sequence to a shared divisible canvas size."""
        sample = (first_input,) + rest_inputs if rest_inputs else first_input
        imgs, anno, dataset = sample
        if not isinstance(imgs, (list, tuple)) or len(imgs) <= 1:
            return sample

        sizes = [query_size(img) for img in imgs]
        if len(set(sizes)) == 1:
            return sample

        target_h = max(h for h, _ in sizes)
        target_w = max(w for _, w in sizes)
        if self.size_divisibility is not None and self.size_divisibility > 1:
            div = int(self.size_divisibility)
            target_h = int(math.ceil(target_h / div) * div)
            target_w = int(math.ceil(target_w / div) * div)

        aligned_imgs = []
        aligned_boxes = []
        for frame_idx, img in enumerate(imgs):
            cur_h, cur_w = sizes[frame_idx]
            pad_right = target_w - cur_w
            pad_bottom = target_h - cur_h
            if pad_right < 0 or pad_bottom < 0:
                raise ValueError(
                    f"Unexpected negative padding at frame {frame_idx}: "
                    f"target=({target_h}, {target_w}), current=({cur_h}, {cur_w})"
                )

            if pad_right == 0 and pad_bottom == 0:
                aligned_imgs.append(img)
                aligned_boxes.append(anno["bbox"][frame_idx])
                continue

            padding = [0, 0, pad_right, pad_bottom]
            aligned_imgs.append(F.pad(img, padding=padding, fill=0))
            aligned_boxes.append(F.pad(anno["bbox"][frame_idx], padding=padding, fill=0))

        anno["bbox"] = aligned_boxes
        return aligned_imgs, anno, dataset


class RandomResize(v2.Transform):

    def __init__(self,
                 scales: list[int],
                 max_size: int = 1536,
                 size_divisibility: int | None = 32,
                 interpolation: Union[F.InterpolationMode,
                 int] = F.InterpolationMode.BILINEAR,
                 antialias: Optional[bool] = True) -> None:
        super().__init__()
        self.size_div = size_divisibility
        scales = [
            i // self.size_div * self.size_div for i in scales
        ] if self.size_div is not None else scales
        self.scales = scales
        self.max_size = max_size // self.size_div * self.size_div if self.size_div is not None else max_size
        self.interpolation = interpolation
        self.antialias = antialias
        self.size_idx = 7

    def set_size_idx(self, idx: int) -> None:
        if len(self.scales) == 0:
            self.size_idx = 0
            return
        self.size_idx = max(0, min(idx, len(self.scales) - 1))

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        min_size = self.scales[self.size_idx]
        h, w = flat_inputs[0].shape[-2:]
        if h > w:
            new_w = min_size
            new_h = (new_w * h / w) // self.size_div * self.size_div
            if new_h > self.max_size:
                new_h = self.max_size
                new_w = new_h * w / h // self.size_div * self.size_div
        else:
            new_h = min_size
            new_w = (new_h * w / h) // self.size_div * self.size_div
            if new_w > self.max_size:
                new_w = self.max_size
                new_h = new_w * h / w // self.size_div * self.size_div

        return {"size": (int(new_h), int(new_w)), "max_size": None}

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._call_kernel(
            F.resize,
            inpt,
            params["size"],
            interpolation=self.interpolation,
            max_size=params["max_size"],
            antialias=self.antialias,
        )


class ResizePadToCanvas(v2.Transform):

    def __init__(
        self,
        canvas_size: tuple[int, int],
        interpolation: Union[F.InterpolationMode, int] = F.InterpolationMode.BILINEAR,
        antialias: Optional[bool] = True,
        fill: float = 0.0,
    ) -> None:
        super().__init__()
        self.canvas_size = canvas_size
        self.interpolation = interpolation
        self.antialias = antialias
        self.fill = fill

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        target_h, target_w = self.canvas_size
        src_h, src_w = query_size(flat_inputs[0])
        scale = min(target_h / max(src_h, 1), target_w / max(src_w, 1))
        new_h = max(1, min(target_h, int(round(src_h * scale))))
        new_w = max(1, min(target_w, int(round(src_w * scale))))
        pad_h = target_h - new_h
        pad_w = target_w - new_w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        return {
            "resize_size": (new_h, new_w),
            "padding": [left, top, right, bottom],
        }

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        resized = self._call_kernel(
            F.resize,
            inpt,
            params["resize_size"],
            interpolation=self.interpolation,
            max_size=None,
            antialias=self.antialias,
        )
        return self._call_kernel(
            F.pad,
            resized,
            padding=params["padding"],
            fill=self.fill,
        )


def transformsV2(mode: str = 'train') -> MultiComposeV2:
    scales = [608, 640, 672, 704, 736, 768, 768, 768, 800, 832, 864, 896, 928]
    if mode == 'train':
        transforms = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            # !: Current video clips and image pseudo-clips keep same-size frames within each sample.
            # Keep this disabled to match the 0401 transform behavior; re-enable only for mixed-size frame sequences.
            # AlignFrameSequenceSize(size_divisibility=32),
            
            v2.RandomPhotometricDistort(p=0.5),
            DynamicRandomSelect(
                transform1=[
                    v2.RandomZoomOut(side_range=(1.0, 3.0), fill=0),
                    MultiRandomIoUCrop(p=0.8, visible_ratio=0.4),
                    MultiSanitizeBoundingBoxes(),  # v2.SanitizeBoundingBoxes(),
                    # !:Keep 0401 DanceTrack behavior: stretch crops to the fixed canvas without preserving aspect ratio.
                    v2.Resize(size=(768, 1365)),
                ],
                transform2=[
                    v2.Resize(size=(768, 1365)),
                ],
                p=1.0,
            ),
            v2.RandomHorizontalFlip(),
            # ResizePadToCanvas(canvas_size=(768, 1365)),  # Preserve-ratio padding variant for later ablation.
            MultiSanitizeBoundingBoxes(),  # v2.SanitizeBoundingBoxes(),
            RandomResize(scales=scales),  # for Multi-Scale
            ConvertBoxes(fmt=BoundingBoxFormat.CXCYWH, normalize=True),
            # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        policy = {"name": "default"}
    else:
        transforms = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            RandomResize(scales=[768]),
            MultiSanitizeBoundingBoxes(),
            ConvertBoxes(fmt=BoundingBoxFormat.CXCYWH, normalize=True),
        ]
        policy = {"name": 'default'}
    return MultiComposeV2(transforms=transforms, policy=policy)


# for debug
def show(sample):
    s = []
    for i in range(len(sample[0])):
        s.append((sample[0][i], sample[1]['bbox'][i], sample[1]['ids'][i],
                  sample[1]['labels'][i]))
    plot(s)
    pass


def plot(imgs, row_title=None, bbox_width=3, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])

    # Process all images and get their original dimensions
    processed_imgs = []
    row_heights = []
    row_widths = []

    for row_idx, row in enumerate(imgs):
        row_images = []
        row_max_height = 0
        row_total_width = 0
        for col_idx, img in enumerate(row):
            boxes, ids, labels, masks = None, None, None, None
            if isinstance(img, tuple):
                img, target, ids, labels = img
                h, w = img.shape[-2:]
                target = target * torch.tensor([w, h, w, h])
                boxes = BoundingBoxes(data=target,  # type: ignore
                                      format="CXCYWH",
                                      canvas_size=(h, w))
                boxes = F.convert_bounding_box_format(
                    inpt=boxes, new_format=BoundingBoxFormat.XYXY)

                labels = [
                    f'{_label.item()}-{_id.item()}'
                    for _id, _label in zip(ids, labels)
                ]
            img = v2.functional.to_image(img)
            if img.dtype.is_floating_point and img.min() < 0:
                # Poor man's re-normalization for the colors to be OK-ish. This
                # is useful for images coming out of Normalize()
                img -= img.min()
                img /= img.max()

            img = v2.functional.to_dtype(img, torch.uint8, scale=True)
            if boxes is not None:
                img = draw_bounding_boxes(img,
                                          boxes,
                                          labels=labels,
                                          colors="yellow",
                                          width=bbox_width)
            if masks is not None:
                img = draw_segmentation_masks(img,
                                              masks.to(torch.bool),
                                              colors=["green"] *
                                                     masks.shape[0],
                                              alpha=.65)

            # Convert to numpy array (H, W, C format)
            img_np = img.permute(1, 2, 0).numpy()
            row_images.append(img_np)

            # Track dimensions: each row's max height and total width
            h, w = img_np.shape[:2]
            row_max_height = max(row_max_height, h)
            row_total_width += w

        processed_imgs.append(row_images)
        row_heights.append(row_max_height)
        row_widths.append(row_total_width)

    # Calculate total dimensions: sum of max heights per row, max of all row widths
    total_height = sum(row_heights)
    total_width = max(row_widths) if row_widths else 0

    # Create canvas with appropriate size
    canvas = torch.zeros((3, total_height, total_width), dtype=torch.uint8)

    # Place images on canvas at original resolution
    current_y = 0
    for row_idx, row_images in enumerate(processed_imgs):
        current_x = 0
        for col_idx, img_np in enumerate(row_images):
            h, w = img_np.shape[:2]
            img_tensor = torch.from_numpy(img_np).permute(
                2, 0, 1)  # Convert to (C, H, W)
            # Place image at current position, respecting original dimensions
            canvas[:, current_y:current_y + h,
            current_x:current_x + w] = img_tensor
            current_x += w
        current_y += row_heights[row_idx]

    # Display the concatenated image
    fig, ax = plt.subplots(1,
                           1,
                           figsize=(total_width / 100, total_height / 100))
    ax.imshow(canvas.permute(1, 2, 0).numpy(), **imshow_kwargs)
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    ax.axis('off')

    plt.tight_layout()
    plt.show()
    pass
