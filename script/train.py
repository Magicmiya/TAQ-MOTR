import os
import time
import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP

# internally
from utils import (
    set_seed,
    Logger,
    MetricLog,
    ProgressLogger,
    is_dist,
    dist_rank,
    dist_world_size,
    is_main_process,
    show_params_name,
    build_stage_policy,
    resolve_stage,
    apply_stage_lr,
    update_best_checkpoints,
    get_param_groups,
    start_manual_save_listener,
    poll_manual_save_command,
    sync_manual_stop,
)
from data import build_dataset, build_dataloader, DataPrefetcher, PrefetchedBatch

from module import build_module, get_model, tensor_list_to_ntensor
from module import copy_checkpoint, load_checkpoint, save_checkpoint, load_pretrained_model
from module import TrackInstances

from .inference import inference_online
from utils.visualizer import GetTime, TensorHook, Visualizer


def train(config: dict):
    train_cfg = config["Training"]
    set_seed(train_cfg["seed"])
    visualizer = None
    vis_result_cfg = None
    if not config["visualize"]:
        TensorHook.deactivate()
        GetTime.deactivate()

    """ # init logger """
    train_logger = Logger(logdir=os.path.join(config["OUTPUTS_DIR"], "train"), only_main=True)
    train_logger.show(head="Training Configs:", log=config, with_header=True)
    train_logger.write(log=config, filename="config.yaml", mode="w")
    train_logger.tb_add_git_version(git_version=config["GIT_VERSION"])

    if config["visualize"]:
        visualizer = Visualizer(cfg=config["Visualizer"], mode="train", save_path=train_logger.logdir)
        infer_cfg = config.get("Inference", config.get("inference", {}))
        lcm_cfg = config.get("Life_cycle_management", {})
        crit_cfg = config.get("Criterion", {})
        result_score_thresh = infer_cfg.get(
            "result_score_thresh",
            lcm_cfg.get("track_thresh", crit_cfg.get("result_score_thresh", None)),
        )
        if result_score_thresh is None:
            result_score_thresh = lcm_cfg.get("track_thresh", crit_cfg.get("track_thresh", None))
        vis_result_cfg = {
            "only_active": bool(infer_cfg.get("result_only_active", False)),
            "min_score": (float(result_score_thresh) if result_score_thresh is not None else None),
            "min_area": (
                float(infer_cfg.get("result_min_area", 100))
                if infer_cfg.get("result_min_area", 100) is not None
                else None
            ),
            "train_interval": int(config.get("Visualizer", {}).get("train_interval", 1)),
        }

    """ # Build Dataset """
    dataset_train = build_dataset(config=config["Dataset"], mode=config["MODE"])

    """ # Build Model"""
    model = build_module(config=config)
    if config["PRETRAINED_MODEL"] is not None:
        model = load_pretrained_model(model, config["PRETRAINED_MODEL"], logger=train_logger, show_details=True)

    """ # Build Optimizer """
    param_groups, lr_names = get_param_groups(config=train_cfg, model=model)
    optimizer = AdamW(
        params=param_groups,
        lr=train_cfg["lr_rate"]["default"],
        weight_decay=train_cfg["weight_decay"],
    )
    stage_policy = build_stage_policy(config)

    """ # Training & State Resume """
    train_states = {
        "start_epoch": 0,
        "start_iter_in_epoch": 0,
        "global_iters": 0,
        "manual_stop": False,
        "last_stage_idx": None,
        "best_eval_metrics": {"HOTA": float("-inf"), "MOTA": float("-inf")},
        "best_eval_epochs": {"HOTA": -1, "MOTA": -1},
    }
    if config["RESUME"] is not None:
        resume_strict = bool(train_cfg.get("resume_strict", True))
        resume_optimizer = bool(train_cfg["resume_scheduler"]) and resume_strict
        # Non-strict resume is a warm-start path for architecture ablations,
        # so optimizer state is skipped to avoid parameter-group mismatches.
        if resume_optimizer:
            load_checkpoint(
                model=model,
                path=config["RESUME"],
                states=train_states,
                optimizer=optimizer,
                scheduler=None,
                strict=resume_strict,
                report_mismatch=True,
            )
        else:
            load_checkpoint(
                model=model,
                path=config["RESUME"],
                states=train_states,
                strict=resume_strict,
                report_mismatch=True,
            )
        train_states.setdefault("last_stage_idx", None)
        train_states.setdefault("best_eval_metrics", {"HOTA": float("-inf"), "MOTA": float("-inf")})
        train_states.setdefault("best_eval_epochs", {"HOTA": -1, "MOTA": -1})
        train_logger.show(head="[Resume] ", log=f"training resume from {config['RESUME']}", write=True)
        train_logger.show(head="[Resume] ", log=f"train_states: {train_states}", write=True)
    start_epoch = int(train_states["start_epoch"])
    start_iter_in_epoch = int(train_states.get("start_iter_in_epoch", 0))
    manual_save_controller, save_command = start_manual_save_listener(train_logger)
    checkpoint_last_name = "checkpoint_last.pth"
    checkpoint_last_path = os.path.join(config["OUTPUTS_DIR"], checkpoint_last_name)

    """ # Debug code """
    # show_params_name(
    #     model,
    #     [100, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310,
    #      311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324])

    """ # Distributed setting """
    if is_dist():
        model = DDP(module=model, device_ids=[dist_rank()], find_unused_parameters=False)

    """
    ========================================
                Training begin ^_^          
    ========================================
    """
    for epoch in range(start_epoch, train_cfg["epochs"]):
        epoch_start_iter = start_iter_in_epoch if epoch == start_epoch else 0
        dataset_train.set_epoch(epoch)
        stage_idx, stage_cfg = resolve_stage(stage_policy=stage_policy, epoch=epoch)
        dataset_train.set_stage(stage_cfg=stage_cfg)
        stage_changes = get_model(model).set_stage(stage_cfg=stage_cfg)
        apply_stage_lr(optimizer=optimizer, lr_names=lr_names, stage_cfg=stage_cfg)
        if train_states.get("last_stage_idx") != stage_idx:
            train_states["last_stage_idx"] = stage_idx
            train_logger.show(
                head="[Stage] ",
                log=f"stage_idx={stage_idx}, stage_cfg={stage_cfg}, applied={stage_changes}",
                write=True,
            )
        dataloader_train = build_dataloader(
            dataset=dataset_train,
            config=config["Dataset"],
            skip_batches=epoch_start_iter,
            seed=train_cfg["seed"],
        )
        pre_fetcher = DataPrefetcher(dataloader_train)

        """ Learning rate Control """
        if epoch >= train_cfg["only_query_updater"]:
            optimizer.param_groups[0]["lr"] = 0.0
            optimizer.param_groups[1]["lr"] = 0.0
            optimizer.param_groups[3]["lr"] = 0.0
        lrs = [optimizer.param_groups[_]["lr"] for _ in range(len(optimizer.param_groups))]
        assert len(lrs) == len(lr_names)
        lr_info = [{name: lr} for name, lr in zip(lr_names, lrs)]
        default_lr_idx = lr_names.index("lr") if "lr" in lr_names else -1
        train_logger.tb_add_scalar(tag="lr", scalar_value=lrs[default_lr_idx], global_step=epoch, mode="epochs")
        for name, lr in zip(lr_names, lrs):
            train_logger.tb_add_scalar(tag=f"lr_{name}", scalar_value=lr, global_step=epoch, mode="epochs")

        """ get no_grad_frames """
        no_grad_frames = None  # for only training query updater
        if len(train_cfg["no_grad_frames"]) > 0:
            assert len(train_cfg["no_grad_frames"]) == len(train_cfg["no_grad_epochs"]), (
                f"no_grad_frames length {len(train_cfg['no_grad_frames'])} "
                f"does not match no_grad_epochs length {len(train_cfg[''])}"
            )
            for i in range(len(config["no_grad_epochs"])):
                if epoch >= config["no_grad_epochs"][i]:
                    no_grad_frames = config["no_grad_frames"][i]
                    break

        """ check_point training to avoid OOM, need change! for your own hardware """
        if config["CHECKPOINT_LEVEL"] == -1:
            frame_num = dataset_train.batch_size * dataset_train.sample_length
            if frame_num > 12:
                get_model(model).checkpoint_level = 2
            elif frame_num > 7:
                get_model(model).checkpoint_level = 1
            else:
                get_model(model).checkpoint_level = 0

        train_logger.epoch_update(
            epoch,
            lr_info,
            dataset_train.batch_size,
            dataset_train.sample_length,
            get_model(model).checkpoint_level,
        )

        epoch_result = train_one_epoch(
            dataloader=pre_fetcher,
            model=model,
            optimizer=optimizer,
            logger=train_logger,
            train_states=train_states,
            max_norm=train_cfg["clip_max_norm"],
            accumulation_steps=train_cfg["accumulation_steps"],
            epoch=epoch,
            multi_checkpoint=train_cfg["multi_checkpoint"],
            no_grad_frames=no_grad_frames,
            visualizer=visualizer,
            vis_result_cfg=vis_result_cfg,
            start_iter=epoch_start_iter,
            max_train_iters=train_cfg.get("max_train_iters", None),
            manual_save_controller=manual_save_controller,
            save_command=save_command,
            tb_loss_log_level=train_cfg.get("tb_loss_log_level", None),
        )
        manual_stop = bool(epoch_result["manual_stop"])
        train_cap_reached = bool(epoch_result.get("train_cap_reached", False))
        next_iter = int(epoch_result["next_iter"])
        dataloader_len = int(epoch_result["dataloader_len"])

        # Handle manual save-and-exit
        if manual_stop:
            if next_iter >= dataloader_len:
                train_states["start_epoch"] = epoch + 1
                train_states["start_iter_in_epoch"] = 0
            else:
                train_states["start_epoch"] = epoch
                train_states["start_iter_in_epoch"] = next_iter
            train_states["manual_stop"] = True
            save_checkpoint(
                model=model,
                path=checkpoint_last_path,
                states=train_states,
                optimizer=optimizer,
                scheduler=None,
                backup=False,
            )
            if is_main_process():
                train_logger.show(
                    head="[Manual Save] ",
                    log=(
                        f"saved checkpoint_last.pth at epoch={epoch}, "
                        f"resume_epoch={train_states['start_epoch']}, "
                        f"resume_iter={train_states['start_iter_in_epoch']} and exiting."
                    ),
                    write=True,
                )
            break

        if train_cap_reached:
            train_logger.show(
                head="[Train Cap] ",
                log=f"stop training after reaching max_train_iters={train_cfg.get('max_train_iters')}.",
                write=True,
            )
            break

        train_states["start_epoch"] = epoch + 1
        train_states["start_iter_in_epoch"] = 0
        train_states["manual_stop"] = False

        # Checkpointing policy:
        # - Maintain a single "last" checkpoint updated every epoch.
        # - Derive long-term checkpoints from "last" without another torch.save.
        save_checkpoint(
            model=model,
            path=checkpoint_last_path,
            states=train_states,
            optimizer=optimizer,
            scheduler=None,
            backup=False,
        )
        if train_cfg["multi_checkpoint"]:
            copy_checkpoint(
                root_dir=config["OUTPUTS_DIR"],
                src_name=checkpoint_last_name,
                dst_name=f"checkpoint_{epoch}.pth",
                logger=train_logger,
            )

        if train_cfg["eval_after_epoch"] > 0 and epoch % train_cfg["eval_after_epoch"] == 0:
            summary = inference_online(
                config=config,
                dataset=dataset_train,
                model=model,
                logger=train_logger,
                epoch=epoch,
            )
            update_best_checkpoints(
                config=config,
                logger=train_logger,
                train_states=train_states,
                summary=summary,
                epoch=epoch,
            )

    copy_checkpoint(wait=True)
    if visualizer is not None:
        visualizer.close()
    return


def train_one_epoch(
    model: nn.Module,
    train_states: dict,
    max_norm: float,
    dataloader: DataPrefetcher,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    logger: Logger,
    accumulation_steps: int = 1,
    use_dab: bool = False,
    multi_checkpoint: bool = False,
    no_grad_frames: int | None = None,
    visualizer: Visualizer | None = None,
    vis_result_cfg: dict | None = None,
    start_iter: int = 0,
    max_train_iters: int | None = None,
    manual_save_controller: dict | None = None,
    save_command: str = "save",
    tb_loss_log_level: int | str | None = None,
):
    model.train()
    optimizer.zero_grad()
    device = next(get_model(model).parameters()).device

    dataloader_len = len(dataloader)
    metric_log = MetricLog(tb_loss_log_level=tb_loss_log_level)
    epoch_start_timestamp = time.time()

    # Initialize progress logger for this epoch
    progress_logger = ProgressLogger(total_len=dataloader_len, head=f"Epoch {epoch}", only_main=True)
    if start_iter > 0 and is_main_process():
        logger.show(
            head="[Resume] ",
            log=f"resume epoch={epoch} from iter={start_iter} (sampler-level batch skipping enabled).",
            write=True,
        )
        progress_logger.update(step_len=start_iter)
    manual_stop = False
    train_cap_reached = False
    next_iter = start_iter
    for i, batch in enumerate(dataloader):
        if max_train_iters is not None and train_states["global_iters"] >= int(max_train_iters):
            train_cap_reached = True
            next_iter = start_iter + i
            break
        iter_idx = start_iter + i

        iter_start_timestamp = time.time()

        tracks = TrackInstances.init_tracks(
            batch=batch,
            hidden_dim=get_model(model).hidden_dim,
            num_classes=get_model(model).num_classes,
            device=device,
        )
        get_model(model).criterion.init_a_clip(batch=batch, device=device)
        get_model(model).life_cycle.init_a_clip(batch=batch, device=device)

        for frame_idx, batch_frame in enumerate(batch):
            imgs = batch_frame.imgs
            frames = tensor_list_to_ntensor(imgs)
            if frames.tensors.device != device:
                frames = frames.to(device)
            if no_grad_frames is None or frame_idx >= no_grad_frames:
                tracks = model(
                    frames=frames,
                    infos=batch_frame.infos[frame_idx],
                    tracks=tracks,
                    batch_data=batch,
                )
            else:
                with torch.no_grad():
                    tracks = model(
                        frames=frames,
                        infos=batch_frame.infos[frame_idx],
                        tracks=tracks,
                        batch_data=batch,
                    )

            if visualizer is not None:
                interval = 1
                only_active = False
                min_score = None
                min_area = 100.0
                if isinstance(vis_result_cfg, dict):
                    interval = int(vis_result_cfg.get("train_interval", 1))
                    only_active = bool(vis_result_cfg.get("only_active", False))
                    min_score = vis_result_cfg.get("min_score", None)
                    min_area = vis_result_cfg.get("min_area", 100.0)
                if interval > 0 and (iter_idx % interval == 0) and len(tracks) > 0:
                    _, _, img_h, img_w = batch_frame.imgs.shape
                    frame_result = tracks[0].result(
                        only_active=only_active,
                        min_score=min_score,
                        min_area=min_area,
                        frame_width=img_w,
                        frame_height=img_h,
                    )
                    visualizer.update(
                        batch=batch_frame,
                        track_result=frame_result,
                        tag=f"e{epoch:03d}_it{train_states['global_iters']:08d}_t{frame_idx:02d}",
                    )

        # Standard batch-level loss calculation (when frame_grad_accumulation is disabled)
        loss, log_dict = get_model(model).criterion.get_normed_loss()
        metric_log["total_loss"] = loss.item()
        loss = loss / accumulation_steps
        loss.backward()
        """ update parameters """
        if (iter_idx + 1) % accumulation_steps == 0:
            if visualizer is not None:
                visualizer.before_grad_clip(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=train_states["global_iters"],
                )

            if max_norm > 0:
                pre_clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            else:
                pre_clip_norm = visualizer.compute_global_grad_norm(model) if visualizer is not None else None
            if visualizer is not None:
                visualizer.after_grad_clip(
                    pre_clip_norm=pre_clip_norm,
                    max_norm=max_norm,
                    global_step=train_states["global_iters"],
                )
            optimizer.step()
            optimizer.zero_grad()

        """ update logger """
        metric_log.update_detail(log_dict)
        iter_end_timestamp = time.time()
        metric_log["time per iter"] = iter_end_timestamp - iter_start_timestamp

        # Outputs logs
        progress_logger.update(step_len=1)
        if iter_idx % (100 // batch.batch_size()) == 0:
            metric_log.sync()
            max_memory = max(
                [torch.cuda.max_memory_allocated(torch.device("cuda", i)) for i in range(dist_world_size())]
            ) // (1024**2)
            second_per_iter = metric_log["time per iter"].avg
            logger.show(
                head=f"[Epoch={epoch}, Iter={iter_idx}/{dataloader_len}, {second_per_iter:.2f}s/iter, "
                f"rest time: {int(second_per_iter * (dataloader_len - iter_idx) // 60)} min, "
                f"Max Memory={max_memory}MB] ",
                log=metric_log,
            )
            logger.write(
                head=f"[Epoch={epoch}, Iter={iter_idx}/{dataloader_len}]",
                log=metric_log,
                filename="log.txt",
                mode="a",
            )
            logger.tb_add_metric_log(log=metric_log, steps=train_states["global_iters"], mode="iters")

        train_states["global_iters"] += 1
        if max_train_iters is not None and train_states["global_iters"] >= int(max_train_iters):
            train_cap_reached = True
            next_iter = iter_idx + 1
            break

        local_stop = False
        if manual_save_controller is not None and is_main_process():
            local_stop = poll_manual_save_command(manual_save_controller)
        if sync_manual_stop(local_stop=local_stop, device=device):
            #  : only stop at batch boundary after loss/optimizer/logger are complete.
            manual_stop = True
            next_iter = iter_idx + 1
            if is_main_process() and local_stop:
                logger.show(
                    head="[Manual Save] ",
                    log=f"received '{save_command}' at epoch={epoch}, iter={iter_idx}; saving checkpoint and exiting.",
                    write=True,
                )
            break

        # Update progress bar with current metrics

    # Epoch end
    metric_log.sync()
    epoch_end_timestamp = time.time()
    epoch_minutes = int((epoch_end_timestamp - epoch_start_timestamp) // 60)
    logger.show(head=f"[Epoch: {epoch}, Total Time: {epoch_minutes}min]", log=metric_log)
    logger.write(
        head=f"[Epoch: {epoch}, Total Time: {epoch_minutes}min]",
        log=metric_log,
        filename="log.txt",
        mode="a",
    )
    logger.tb_add_metric_log(log=metric_log, steps=epoch, mode="epochs")

    return {
        "manual_stop": manual_stop,
        "train_cap_reached": train_cap_reached,
        "next_iter": next_iter,
        "dataloader_len": dataloader_len,
    }
