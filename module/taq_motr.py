import os
import torch
import torch.nn as nn
import inspect

from torch.utils.checkpoint import checkpoint
from typing import List, cast

from .nn import build_backbone, build_encoder, build_decoder
from .nn.query_updater import build as build_query_updater
from .nn.life_cycle_management import build as build_life_cycle_management
from .nn.criterion import build as build_criterion
from .nn.instance import TrackInstances
from .nn import NestedTensor
from utils.visualizer import GetTime
from data import PrefetchedBatch


class TAQ_MOTR(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        criterion: nn.Module,
        query_updater: nn.Module,
        life_cycle: nn.Module,
        hidden_dim=256,
        num_classes=1,
        checkpoint_level=0,
        visualize=False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.checkpoint_level = checkpoint_level
        self.visualize = visualize
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.criterion = criterion
        self.query_updater = query_updater
        self.life_cycle = life_cycle
        if self.visualize:
            os.makedirs("./outputs/visualize_tmp/taq_motr/", exist_ok=True)

    def set_stage(self, stage_cfg: dict | None = None) -> dict[str, object]:
        stage_cfg = stage_cfg or {}
        applied: dict[str, object] = {}

        if "use_aux_loss" in stage_cfg:
            use_aux_loss = bool(stage_cfg["use_aux_loss"])
            self.decoder.aux_loss = use_aux_loss
            self.criterion.aux_loss = use_aux_loss
            applied["use_aux_loss"] = use_aux_loss

        if "use_det_dn_aux" in stage_cfg:
            use_det_dn_aux = bool(stage_cfg["use_det_dn_aux"])
            self.decoder.det_dn_aux_last_only = not use_det_dn_aux
            applied["use_det_dn_aux"] = use_det_dn_aux

        if "high_conf_threshold" in stage_cfg:
            high_conf_threshold = float(stage_cfg["high_conf_threshold"])
            self.life_cycle.high_conf_threshold = high_conf_threshold
            applied["high_conf_threshold"] = high_conf_threshold

        if "use_dn" in stage_cfg:
            use_dn = bool(stage_cfg["use_dn"])
            query_generator = self.decoder.query_generator
            dn_stopped = bool(getattr(query_generator, "_dn_stopped", False))
            if use_dn:
                if dn_stopped:
                    raise ValueError("Stage policy tried to re-enable DN after stopDN(); this is not supported.")
            else:
                query_generator.stopDN()
            applied["use_dn"] = use_dn

        return applied

    @GetTime("Model_forward")
    def forward(
        self, frames: NestedTensor, infos: List[dict], tracks: list[TrackInstances], batch_data: PrefetchedBatch
    ):
        """Backbone"""
        if self.checkpoint_level >= 3:
            f_backbone = checkpoint(self.backbone, frames, use_reentrant=False)
        else:
            f_backbone = self.backbone(frames)

        """ Encoder """
        if self.checkpoint_level >= 1:
            _res = checkpoint(self.encoder, f_backbone, frames.masks, use_reentrant=False)
            feature_encoder, feature_mask = cast(tuple[list[torch.Tensor], list[torch.Tensor]], _res)
        else:
            feature_encoder, feature_mask = self.encoder(f_backbone, frames.masks)

        """ Query_updater """
        tracks = self.query_updater(tracks)

        """ Decoder """
        decoder_output = self.decoder(feature_encoder, feature_mask, infos, tracks, self.checkpoint_level)

        """ Life-cycle management """
        batch_data.preload()
        if not self.training:
            return self.life_cycle(tracks, decoder_output)

        """ Criterion """
        tracks, det_matched = self.criterion(decoder_output, tracks)
        return self.life_cycle(tracks, decoder_output, det_matched=det_matched)


def build_module(config: dict):
    # get module config
    sig = inspect.signature(TAQ_MOTR)
    _cfg = {k: v for k, v in config.items() if k in sig.parameters}

    return TAQ_MOTR(
        backbone=build_backbone(config=config),
        encoder=build_encoder(config=config),
        decoder=build_decoder(config=config),
        criterion=build_criterion(config=config),
        query_updater=build_query_updater(config=config),
        life_cycle=build_life_cycle_management(config=config),
        checkpoint_level=config["CHECKPOINT_LEVEL"],
        **_cfg,
    )
