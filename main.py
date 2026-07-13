import os
import torch.distributed
import torch.backends.cuda
import torch.backends.cudnn
from configs import get_config
from configs.utils import sync_distributed_train_output_dir
from datetime import timedelta


def _init_distributed_once(config: dict):
    os.environ["CUDA_VISIBLE_DEVICES"] = config["Training"]["Available_gpus"]
    from utils.utils import dist_rank

    if config["Training"]["use_distributed"]:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl", timeout=timedelta(hours=1))
        torch.cuda.set_device(dist_rank())
        config = sync_distributed_train_output_dir(config)
    return config


def main(config: dict):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = _init_distributed_once(config)

    if config["MODE"] == "train":
        from script import train

        train(config=config)
    elif config["MODE"] == "val":
        from script.inference import inference_offline

        inference_offline(config=config)
    elif config["MODE"] == "test":
        from script.inference import inference_offline

        inference_offline(config=config)
    else:
        raise ValueError(f"Unsupported mode '{config['MODE']}'")
    return


if __name__ == "__main__":
    cfg_list = get_config()
    for cfg in cfg_list:
        main(config=cfg)
