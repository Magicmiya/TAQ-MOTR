# @Author       : Ruopeng Gao
# @Date         : 2022/7/5
# @Description  : Logger will log information.
import os
import json
import argparse
import yaml
from datetime import datetime

from tqdm import tqdm, trange
from typing import Any
from torch.utils import tensorboard as tb

from utils.log.log import MetricLog
from utils.utils import is_main_process, dist_rank, is_dist, dist_world_size
from rich.console import Console
from rich.pretty import Pretty


class ProgressLogger:
    def __init__(self, total_len: int, head: str = None, only_main: bool = True):
        self.only_main = only_main
        self.rank = dist_rank() if is_dist() else 0
        self.world_size = dist_world_size() if is_dist() else 1
        if (self.only_main and is_main_process()) or (self.only_main is False):
            self.total_len = total_len
            self.tqdm = tqdm(total=total_len, ncols=100, mininterval=1, position=self.rank)
            self.head = f"[GPU {self.rank}] {head}" if head else f"GPU {self.rank}"
            #
        else:
            self.total_len = total_len
            self.tqdm = None
            self.head = head

    def update(self, step_len: int, **kwargs: Any):
        if self.only_main and not is_main_process():
            return
        else:
            self.tqdm.set_description(self.head)
            self.tqdm.set_postfix(**kwargs)
            self.tqdm.update(step_len)

    def set_head(self, head: str):
        if (self.only_main and is_main_process()) or (self.only_main is False):
            self.head = head
        else:
            self.head = f'Rank-{dist_rank()} {head} '


class Logger:
    """
    Log information.
    """

    def __init__(self, logdir: str, only_main: bool = True, enable_tensorboard: bool = True):
        self.only_main = only_main
        self.enable_tensorboard = bool(enable_tensorboard)
        self.console = Console(width=1000, color_system="auto")
        self.tb_iters_logger: tb.SummaryWriter | None = None
        self.tb_epochs_logger: tb.SummaryWriter | None = None
        self.tb_eval_logger: tb.SummaryWriter | None = None
        if self.only_main and not is_main_process():
            self.logdir = logdir
        else:
            self.logdir = logdir
            os.makedirs(self.logdir, exist_ok=True)
            if self.enable_tensorboard:
                self.tb_iters_logger = tb.SummaryWriter(log_dir=os.path.join(self.logdir, "tb_iters_log"))
                self.tb_epochs_logger = tb.SummaryWriter(log_dir=os.path.join(self.logdir, "tb_epochs_log"))
                self.tb_eval_logger = tb.SummaryWriter(log_dir=os.path.join(self.logdir, "tb_eval_log"))

        return

    def show(self, head: str = "", log: str | dict | MetricLog = "", write=False, with_header=False):
        if self.only_main and not is_main_process():
            return
        _head = head if not with_header else "\n" + "=" * 80 + f"\n{head}\n" + "=" * 80 + "\n"
        if isinstance(log, dict):
            self.console.print(f"[bold]{_head}[/bold]", style="green")
            pretty = Pretty(log, indent_guides=True, expand_all=True)
            self.console.print(pretty, style="green")
        elif isinstance(log, MetricLog):
            self.console.print(f"[bold]{_head}[/bold]", end="", style="green")
            self.console.print(log, style="green")
        else:
            self.console.print(f"[bold]{_head}[/bold]", end="", style="green")
            self.console.print(log, style="green")
        if write is True:
            self.write(_head, log)
        return

    def print(self, log):
        self.console.print(log, style="green")

    def write(self, head: str = "", log: dict | str | MetricLog = "", filename: str = "log.txt", mode: str = "a"):
        """
        Logger write a log to a file.

        Args:
            head: Log head like self.show.
            log: A log.
            filename: Write file name.
            mode: Open file with this mode.
        """
        if self.only_main and not is_main_process():
            return
        if isinstance(log, dict):
            if head != "":
                raise Warning("Log is a dict, Do not support 'head' attr.")
            if len(filename) > 5 and filename[-5:] == ".yaml":
                self._write_dict_to_yaml(log, filename, mode)
            elif len(filename) > 5 and filename[-5:] == ".json":
                self._write_dict_to_json(log, filename, mode)
            elif len(filename) > 4 and filename[-4:] == ".txt":
                self._write_dict_to_json(log, filename, mode)
            else:
                raise RuntimeError("Filename '%s' is not supported for dict log." % filename)
        elif isinstance(log, MetricLog):
            with open(os.path.join(self.logdir, filename), mode=mode) as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                f.write(f"[{timestamp}] {head} {log}\n")
        elif isinstance(log, str):
            with open(os.path.join(self.logdir, filename), mode=mode) as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                f.write(f"[{timestamp}] {head}{log}\n")
        else:
            raise RuntimeError("Log type '%s' is not supported." % type(log))
        return

    def _write_dict_to_yaml(self, log: dict, filename: str, mode: str = "w"):
        """
        Logger writes a dict log to a .yaml file.

        Args:
            log: A dict log.
            filename: A yaml file's name.
            mode: Open with this mode.
        """
        with open(os.path.join(self.logdir, filename), mode=mode) as f:
            yaml.safe_dump(log, f, allow_unicode=True, sort_keys=False, default_flow_style=False, indent=2)
        return

    def _write_dict_to_json(self, log: dict, filename: str, mode: str = "w"):
        """
        Logger writes a dict log to a .json file.

        Args:
            log (dict): A dict log.
            filename (str): Log file's name.
            mode (str): File writing mode, "w" or "a".
        """
        with open(os.path.join(self.logdir, filename), mode=mode) as f:
            f.write(json.dumps(log, indent=4))
            f.write("\n")
        return

    def tb_add_scalar(self, tag: str, scalar_value: float, global_step: int, mode: str):
        if self.only_main and not is_main_process():
            return
        if mode == "iters":
            writer = self.tb_iters_logger
        else:
            writer = self.tb_epochs_logger
        if writer is None:
            return
        writer.add_scalar(tag=tag, scalar_value=scalar_value, global_step=global_step)
        return

    def tb_add_metric_log(self, log: MetricLog, steps: int, mode: str):
        if self.only_main and not is_main_process():
            return
        tags, tag_scalar_dicts = log.get_tb(mode)
        if mode == "iters":
            writer = self.tb_iters_logger
        else:
            writer = self.tb_epochs_logger
        if writer is None:
            return
        for tag, scalar_dict in zip(tags, tag_scalar_dicts):
            writer.add_scalars(main_tag=tag, tag_scalar_dict=scalar_dict, global_step=steps)
        return

    def tb_add_git_version(self, git_version: str):
        if self.only_main and not is_main_process():
            return
        git_version = "null" if git_version is None else git_version
        if self.tb_iters_logger is not None:
            self.tb_iters_logger.add_text(tag="git_version", text_string=git_version)
        if self.tb_epochs_logger is not None:
            self.tb_epochs_logger.add_text(tag="git_version", text_string=git_version)
        return

    def epoch_update(self, epoch, lr, batch_size, sample_length, check_point_level):
        if self.only_main and not is_main_process():
            return
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        _head = "\n" + "=" * 80 + f"\n[{timestamp}][Epoch {epoch}] lr={lr}\n" + "=" * 80 + "\n"
        _log = f"batch_size:{batch_size} sample_length: {sample_length} check_point_level={check_point_level}"
        self.show(head=_head, log=_log)
        self.write(head=_head, log=_log)


def parser_to_dict(log: argparse.ArgumentParser) -> dict:
    opts_dict = dict()
    for k, v in vars(log).items():
        if v:
            opts_dict[k] = v
    return opts_dict
