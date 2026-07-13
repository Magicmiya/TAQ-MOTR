import warnings

import torch

from utils import dist_rank

from .nn.instance import TrackInstances
from .nn.common.utils import *
from .nn.common.nested_tensor import NestedTensor, tensor_list_to_ntensor


def build_module(config: dict):
    module_name = config["MODULE_NAME"].lower()
    if module_name == "taq_motr":
        from .taq_motr import build_module as build_taq_motr
    else:
        raise ValueError(f"Unsupported module '{config['MODULE_NAME']}'")
    model = build_taq_motr(config=config)
    training_cfg = config["Training"]
    if training_cfg["Available_gpus"] is not None and training_cfg["device"] == "cuda":
        model.to(device=torch.device(training_cfg["device"], dist_rank()))
    else:
        model.to(device=torch.device(config["device"]))
    return model
