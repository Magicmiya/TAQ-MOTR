# @Author       : Hanyang Liu
# @Date         : 2025/7/8
# @Description  : NestedTensor，modified from MOTR/Deformable DETR
# Modified from MeMOTR (https://github.com/MCG-NJU/MeMOTR)
import torch
from typing import Optional, List


class NestedTensor(object):
    def __init__(self, tensors: torch.Tensor, masks: Optional[torch.Tensor]):
        """
        Args:
            tensors: Tensor, (B, C, H, W)
            masks: Tensor, (B, H, W)
        """
        if masks is not None:
            assert tensors.shape[0] == masks.shape[0], \
                f"tensors have batch size {tensors.shape[0]} but get {masks.shape[0]} for mask."
        self.tensors = tensors
        self.masks = masks

    def to(self, device, non_blocking=False):
        """
        Args:
            device:
            non_blocking:
        """
        tensors = self.tensors.to(device, non_blocking=non_blocking)
        if self.masks is None:
            masks = None
        else:
            masks = self.masks.to(device, non_blocking=non_blocking)
        return NestedTensor(tensors=tensors, masks=masks)

    def decompose(self) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.tensors, self.masks

    def __repr__(self):
        return str(self.tensors)


def tensor_list_to_ntensor(
        inputs: List[torch.Tensor] | torch.Tensor,
        size_divisibility: int = 32) -> NestedTensor:
    if isinstance(inputs, torch.Tensor):
        b, c, h, w = inputs.shape
        masks = torch.zeros((b, h, w), dtype=torch.bool, device=inputs.device)
        return NestedTensor(tensors=inputs, masks=masks)

    assert inputs[0].dim() == 3, f"Tensor should have 3 dimensions, but get {inputs[0].dim()}"
    heights, widths = zip(*[t.shape[1:] for t in inputs])
    final_shape = [len(inputs)] + [inputs[0].shape[0]] + list(map(max, (heights, widths)))
    final_b, final_c, final_h, final_w = final_shape
    if size_divisibility > 0:
        stride = size_divisibility
        final_h = (final_h + (stride - 1)) // stride * stride
        final_w = (final_w + (stride - 1)) // stride * stride
    final_shape = [final_b, final_c, final_h, final_w]
    dtype = inputs[0].dtype
    device = inputs[0].device
    tensors = torch.zeros(final_shape, dtype=dtype, device=device)
    masks = torch.ones((final_b, final_h, final_w), dtype=torch.bool, device=device)
    for input_tensor, pad_tensor, mask in zip(inputs, tensors, masks):
        assert input_tensor.shape[0] == final_shape[1], "Tensor channel size should be equal."
        pad_tensor[: input_tensor.shape[0], : input_tensor.shape[1], : input_tensor.shape[2]].copy_(input_tensor)
        mask[: input_tensor.shape[1], : input_tensor.shape[2]] = False
    return NestedTensor(tensors=tensors, masks=masks)
