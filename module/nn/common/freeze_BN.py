# copy and modified from https://github.com/facebookresearch/detr/blob/master/models/backbone.py


import torch
import torch.nn as nn


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, num_features, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        n = num_features
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps
        self.num_features = n

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

    def extra_repr(self):
        return (
            "{num_features}, eps={eps}".format(**self.__dict__)
        )


def freeze_batch_norm2d(module: nn.Module) -> nn.Module:
    """
    Recursively replace all nn.BatchNorm2d layers with FrozenBatchNorm2d.
    
    This function freezes BatchNorm layers by keeping their statistics (mean and variance) 
    constant during inference, improving performance and reducing features usage.
    
    Args:
        module (nn.Module): Neural network module to process
        
    Returns:
        nn.Module: Processed module with all BatchNorm2d replaced by FrozenBatchNorm2d
    """
    if isinstance(module, nn.BatchNorm2d):
        # Replace BatchNorm2d with FrozenBatchNorm2d
        module = FrozenBatchNorm2d(module.num_features)
    else:
        # Recursively process all child modules
        for name, child in module.named_children():
            _child = freeze_batch_norm2d(child)
            if _child is not child:
                setattr(module, name, _child)
    return module


def no_local_batch_norm2d(module: nn.Module) -> nn.Module:
    # todo: need improve
    return freeze_batch_norm2d(module)
