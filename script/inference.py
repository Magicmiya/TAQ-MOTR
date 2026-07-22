import os
import time
import copy
import shutil
import zipfile
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data import build_dataset, build_dataloader, DataPrefetcher, PrefetchedBatch, MOTDataset
from module import build_module, get_model, tensor_list_to_ntensor, TrackInstances, load_checkpoint
from utils import set_seed, Logger, ProgressLogger, is_dist, dist_rank, dist_world_size, is_main_process
from utils.visualizer import GetTime, TensorHook, Visualizer
from . import evaluator as eval_utils


def _resolve_inference_root(cfg: dict, mode: str) -> str:
    mode_lower = str(mode).lower()
    if mode_lower in {"val", "test"}:
        return os.path.join(cfg["OUTPUTS_DIR"], mode_lower)
    return cfg["OUTPUTS_DIR"]


def _build_eval_sub_out_path(cfg: dict, eval_file_path: str) -> str:
    checkpoint_name = os.path.splitext(os.path.basename(eval_file_path))[0]
    exp_name = str(cfg.get("EXP_NAME", "")).strip()
    if not exp_name:
        return checkpoint_name
    # Keep per-run eval folders unique inside inference_val/inference_test while
    # preserving the user's requested layout under the checkpoint root.
    return f"{exp_name}_{checkpoint_name}"


def _build_submission_zip(result_dir: str, zip_path: str):
    txt_files = sorted([file for file in os.listdir(result_dir) if file.endswith(".txt")])
    if not txt_files:
        raise FileNotFoundError(f"No tracking result txt files found in: {result_dir}")

    # Package submission txt files at zip root so DanceTrack upload format matches the benchmark requirement.
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for file_name in txt_files:
            abs_path = os.path.join(result_dir, file_name)
            zip_file.write(abs_path, arcname=file_name)


def _get_submission_coordinate_offset(dataset_name: str) -> tuple[float, float]:
    dataset_key = str(dataset_name or "").strip().lower()
    if dataset_key == "sportsmot":
        # SportsMOT submission follows 1-based top-left coordinates, so only left/top shift by +1.
        return 1.0, 1.0
    return 0.0, 0.0


def _replace_detector_suffix(video_name: str, detector: str) -> str:
    prefix, _, _ = video_name.rpartition("-")
    if not prefix:
        raise ValueError(f"Invalid MOTChallenge video name: {video_name}")
    return f"{prefix}-{detector}"


def _stage_motchallenge_submission(dataset: MOTDataset, result_dir: str, run_dir: str) -> str:
    if not hasattr(dataset, "get_submission_layout"):
        raise TypeError("MOTChallenge submission requires dataset submission metadata.")

    layout = dataset.get_submission_layout()
    primary_detector = str(layout.get("primary_detector") or "")
    detectors = tuple(layout["detectors"])
    use_detector_suffix = bool(layout.get("use_detector_suffix", False))
    train_videos = list(layout["train_videos"])
    test_videos = list(layout["test_videos"])

    submission_dir = os.path.join(run_dir, "submission_txt")
    if os.path.isdir(submission_dir):
        shutil.rmtree(submission_dir)
    os.makedirs(submission_dir, exist_ok=True)

    # Training sequences are still required by MOTChallenge submission packages, so we stage them from GT directly.
    for video_name in train_videos:
        gt_path = os.path.join(dataset.dataset_dir, "train", video_name, "gt", "gt.txt")
        if not os.path.isfile(gt_path):
            raise FileNotFoundError(f"Train GT file not found: {gt_path}")
        if use_detector_suffix:
            for detector in detectors:
                shutil.copy2(gt_path, os.path.join(submission_dir, f"{_replace_detector_suffix(video_name, detector)}.txt"))
        else:
            shutil.copy2(gt_path, os.path.join(submission_dir, f"{video_name}.txt"))

    # MOT17 test inference fans out detector suffixes, while MOT20 keeps one txt per sequence.
    for video_name in test_videos:
        src_path = os.path.join(result_dir, f"{video_name}.txt")
        if not os.path.isfile(src_path):
            raise FileNotFoundError(f"Primary detector result not found: {src_path}")
        if use_detector_suffix:
            for detector in detectors:
                dst_name = _replace_detector_suffix(video_name, detector)
                if detector == primary_detector:
                    shutil.copy2(src_path, os.path.join(submission_dir, f"{dst_name}.txt"))
                else:
                    shutil.copy2(src_path, os.path.join(submission_dir, f"{dst_name}.txt"))
        else:
            shutil.copy2(src_path, os.path.join(submission_dir, f"{video_name}.txt"))

    return submission_dir


def _get_test_run_paths(cfg: dict, sub_out_path: str) -> tuple[str, str, str]:
    infer_root = _resolve_inference_root(cfg=cfg, mode="test")
    run_root = os.path.join(infer_root, "inference_test")
    run_dir = os.path.join(run_root, sub_out_path)
    result_dir = os.path.join(run_dir, "result")
    submission_zip_path = os.path.join(run_root, f"submission_{sub_out_path}.zip")
    return run_dir, result_dir, submission_zip_path


def _prepare_test_run_layout(cfg: dict, sub_out_path: str) -> tuple[str, str, str]:
    run_dir, result_dir, submission_zip_path = _get_test_run_paths(cfg=cfg, sub_out_path=sub_out_path)

    # Reset the per-checkpoint test export directory so reruns always produce a clean submission workspace.
    if os.path.isdir(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(result_dir, exist_ok=True)

    # Keep a blank online-eval log placeholder beside the exported result folder for manual benchmark feedback.
    eval_log_path = os.path.join(run_dir, "eval_log.txt")
    with open(eval_log_path, "w", encoding="utf-8"):
        pass

    # Snapshot the merged runtime config next to the submission assets for exact reproduction.
    config_path = os.path.join(run_dir, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)

    return run_dir, result_dir, submission_zip_path


def _inference_core(
    cfg: dict,
    dataset: MOTDataset,
    model: nn.Module,
    logger: Logger,
    mode: str,
    sub_out_path="",
    visualizer: Visualizer | None = None,
    require_complete_gt: bool = False,
):
    """Inference the dataset through a trained model"""
    if is_dist():
        dist.barrier()
        rank = dist_rank()
        world_size = dist_world_size()
        org_model = get_model(model).eval()
        # Align rank-local buffers such as BatchNorm running stats with the checkpoint-saving rank.
        for buffer in org_model.buffers():
            dist.broadcast(buffer, src=0)
        dist.barrier()
    else:
        rank, world_size = 0, 1
        org_model = model.eval()
    eval_start_timestamp = 0
    if is_main_process():
        eval_start_timestamp = time.time()

    """ build Dataset """
    dataset.eval(mode=mode, rank=rank, world_size=world_size)
    has_complete_gt = False
    if str(mode).lower() == "test":
        has_complete_gt = eval_utils.has_complete_split_gt(dataset=dataset, split=mode)
        if require_complete_gt and not has_complete_gt:
            raise RuntimeError(
                "Training-time test evaluation requires gt/gt.txt for every selected test sequence."
            )
    dataloader_eval = build_dataloader(
        dataset=dataset,
        config=cfg["Dataset"],
        seed=cfg.get("Training", {}).get("seed", 0),
    )
    dataloader_len = len(dataloader_eval)
    pre_fetcher = DataPrefetcher(dataloader_eval)
    device = next(org_model.parameters()).device
    # Snapshot the selected eval subset once so logging and inference iterate the same sequence set.
    selected_videos = dataset.get_selected_videos(mode=mode)

    # Keep split-specific inference artifacts under OUTPUTS_DIR/<mode> so val/test runs do not overwrite each other.
    inference_root = _resolve_inference_root(cfg=cfg, mode=mode)
    if str(mode).lower() == "test":
        out_path, result_dir, _ = _prepare_test_run_layout(cfg=cfg, sub_out_path=sub_out_path)
    else:
        out_path = str(os.path.join(inference_root, f"inference_{mode}", sub_out_path))
        result_dir = out_path
    # Let every rank observe a ready result directory before any worker writes temp result files.
    os.makedirs(result_dir, exist_ok=True)
    if is_dist():
        dist.barrier()
    if is_main_process():
        # build sample video info
        _log = ""
        for _rank in range(world_size):
            _log += f"Rank {_rank}: "
            vid_names = []
            for vid in selected_videos:
                if dataset.vid_idx[vid] % world_size == _rank:
                    vid_names.append(vid)
            _log += f"(Num:{len(vid_names)}) : {','.join(vid_names)}"
            _log += "\n"
        logger.show(
            head=f"inference task - All GPU Data Info",
            log=_log,
            write=True,
            with_header=True,
        )
    # inference progress bar
    progress_logger = ProgressLogger(total_len=dataloader_len, head=f"Inference {mode}", only_main=False)

    """ begin inference """
    infer_cfg = cfg.get("Inference", cfg.get("inference", {})) if isinstance(cfg, dict) else {}
    lcm_cfg = cfg.get("Life_cycle_management", {}) if isinstance(cfg, dict) else {}
    crit_cfg = cfg.get("Criterion", {}) if isinstance(cfg, dict) else {}
    result_score_thresh = infer_cfg.get(
        "result_score_thresh",
        lcm_cfg.get("track_thresh", crit_cfg.get("result_score_thresh", None)),
    )
    if result_score_thresh is None:
        result_score_thresh = lcm_cfg.get("track_thresh", crit_cfg.get("track_thresh", None))
    result_min_area = infer_cfg.get("result_min_area", 100)
    result_only_active = infer_cfg.get("result_only_active", False)

    f_path = ""
    video_name = ""
    track_result = []
    for i, batch in enumerate(pre_fetcher):
        with torch.no_grad():
            _video_name = batch.get_video_name()
            _frame_id = batch.get_frame_idx()

            """ New videos """
            if video_name != _video_name:
                if len(video_name) > 0:
                    write_results(f_path, track_result, dataset_name=getattr(dataset, "name", ""))
                    track_result = []
                video_name = _video_name
                assert (
                    batch.get_frame_idx() == 1
                ), f" video {video_name} should begin with frame 1 but got {batch.get_frame_idx()}"
                f_path = os.path.join(result_dir, f"{video_name}.txt")
                org_model.life_cycle.frame_idx = 0
                org_model.life_cycle.max_obj_id = 1
                org_model.life_cycle.init_a_clip(batch=batch, device=device)
                tracks = TrackInstances.init_tracks(
                    batch=batch,
                    hidden_dim=org_model.hidden_dim,
                    num_classes=org_model.num_classes,
                    device=device,
                )

            """ Inference img """
            batch_frame = batch.next_frame()
            assert batch_frame is not None, "next_frame() returned None unexpectedly during evaluation."
            imgs = batch_frame.imgs
            frames = tensor_list_to_ntensor(imgs)
            tracks = org_model(
                frames=frames,
                infos=batch_frame.infos[0],
                tracks=tracks,
                batch_data=batch,
            )

            """ Save results and visualizer update """
            frame_result = tracks[0].result(
                only_active=bool(result_only_active),
                min_score=(float(result_score_thresh) if result_score_thresh is not None else None),
                min_area=(float(result_min_area) if result_min_area is not None else None),
            )
            track_result.append(frame_result)
            if org_model.visualize and visualizer is not None:
                visualizer.update(batch_frame, frame_result)
            progress_logger.update(step_len=1, **{video_name: f"{_frame_id:0>{4}}"})

            """ Save the last video"""
            if i == dataloader_len - 1:
                write_results(f_path, track_result, dataset_name=getattr(dataset, "name", ""))

    if is_dist():
        dist.barrier()

    logger.show(
        head=f" inference time: ",
        log=f"{time.time() - eval_start_timestamp} sec",
        write=True,
    )
    logger.show(head=f" result files path: ", log=f"{result_dir}", write=True)
    # Clear GPU memory after inference
    torch.cuda.empty_cache()
    return out_path, result_dir, has_complete_gt


def inference_offline(config: dict):
    # Offline engine orchestrates checkpoint loop, core inference, and TrackEval reporting.
    set_seed(config["Training"]["seed"])
    eval_files = []
    if os.path.isfile(config["EVAL_FILE_PATH"]):
        eval_files.append(config["EVAL_FILE_PATH"])
    else:
        eval_files = sorted(
            [
                os.path.join(config["EVAL_FILE_PATH"], f)
                for f in os.listdir(config["EVAL_FILE_PATH"])
                if f.endswith(".pth")
            ]
        )

    infer_mode = str(config.get("MODE", "")).lower() in {"val", "test"}
    eval_logger = Logger(
        logdir=str(os.path.join(config["OUTPUTS_DIR"], config["MODE"])),
        only_main=True,
        enable_tensorboard=not infer_mode,
    )
    eval_logger.show(
        head="Evaluation files list",
        log="\n".join(eval_files),
        with_header=True,
        write=True,
    )
    eval_logger.tb_add_git_version(git_version=config["GIT_VERSION"])
    eval_logger.show(head="Training Configs:", log=config, with_header=True)

    dataset_eval = build_dataset(config=config["Dataset"], mode=config["MODE"])

    for file in eval_files:
        if is_dist():
            dist.barrier()

        infer_root = _resolve_inference_root(cfg=config, mode=config["MODE"])
        sub_out_path = _build_eval_sub_out_path(cfg=config, eval_file_path=file)
        if config["visualize"]:
            visualizer_cfg = copy.deepcopy(config["Visualizer"])
            # Route visualize exports through the same run name as inference outputs.
            visualizer_cfg["root_dir_name"] = f"visualize_{sub_out_path}"
            visualizer = Visualizer(
                cfg=visualizer_cfg,
                mode=config["MODE"],
                save_path=infer_root,
            )
        else:
            TensorHook.deactivate()
            GetTime.deactivate()
            visualizer = None

        model = build_module(config=config)
        load_checkpoint(model=model, path=file, strict=True, report_mismatch=True)

        if is_dist():
            model = DDP(
                module=model,
                device_ids=[dist_rank()],
                find_unused_parameters=False,
            )

        trackers_to_eval = f"inference_{config['MODE']}"
        run_dir, result_dir, test_has_complete_gt = _inference_core(
            cfg=config,
            dataset=dataset_eval,
            model=model,
            logger=eval_logger,
            mode=config["MODE"],
            sub_out_path=sub_out_path,
            visualizer=visualizer,
        )
        if visualizer is not None:
            visualizer.close()

        summary = {}
        eval_report = {}
        eval_step = eval_utils._get_eval_step(sub_out_path)
        eval_metric_log = None
        eval_time_cost = None
        should_evaluate = config["MODE"] == "val" or (
            config["MODE"] == "test" and test_has_complete_gt
        )

        if is_main_process() and config["MODE"] == "test":
            run_dir, _, submission_zip_path = _get_test_run_paths(cfg=config, sub_out_path=sub_out_path)
            submission_source_dir = result_dir
            if getattr(dataset_eval, "name", "") in {"MOT17", "MOT20"}:
                submission_source_dir = _stage_motchallenge_submission(
                    dataset=dataset_eval,
                    result_dir=result_dir,
                    run_dir=run_dir,
                )
            _build_submission_zip(result_dir=submission_source_dir, zip_path=submission_zip_path)
            eval_logger.show(
                head=f"Submission package: {sub_out_path}",
                log=f"{submission_zip_path}",
                write=True,
                with_header=True,
            )

        if should_evaluate and is_main_process():
            tracker_sub_folder = sub_out_path
            if config["MODE"] == "test":
                tracker_sub_folder = os.path.join(sub_out_path, "result")
            eval_start_timestamp = time.time()
            summary, eval_report = eval_utils.evaluate(
                config,
                dataset_eval,
                trackers_to_eval,
                tracker_sub_folder,
            )
            eval_time_cost = time.time() - eval_start_timestamp

        summary = eval_utils._broadcast_eval_summary(summary)
        if summary:
            eval_metric_log = eval_utils._build_eval_metric_log(summary)
            eval_metric_log.sync()

        if is_main_process() and should_evaluate:
            eval_msg = ""
            if eval_time_cost is not None:
                eval_msg = f"eval time: {eval_time_cost:<6.2f} seconds\n"
            if summary:
                eval_msg += eval_utils._format_eval_summary(summary)
            else:
                eval_msg += "No summary output, eval failed!"
            eval_logger.show(
                head=f"Evaluation Finish: {sub_out_path}",
                log=eval_msg,
                write=True,
                with_header=True,
            )
            if eval_metric_log is not None and eval_step is not None:
                eval_utils.metrics_to_tensorboard(eval_logger.tb_eval_logger, eval_metric_log, eval_step)
            eval_utils.write_default_eval_result(
                config=config,
                eval_split=config["MODE"],
                trackers_to_eval=trackers_to_eval,
                sub_out_path=sub_out_path,
                eval_msg=eval_msg,
                trackeval_stdout=str(eval_report.get("stdout", "")),
                formatted_tsv=str(eval_report.get("formatted_tsv", "")),
            )
            eval_utils.copy_eval_result_to_visualize(
                config=config,
                eval_split=config["MODE"],
                trackers_to_eval=trackers_to_eval,
                sub_out_path=sub_out_path,
            )
        del model
        torch.cuda.empty_cache()
        if is_dist():
            dist.barrier()


def inference_online(
    config: dict,
    dataset: MOTDataset,
    model: nn.Module,
    logger: Logger,
    epoch: int,
    eval_split: str = "val",
    use_tb=True,
):
    eval_split = str(eval_split).strip().lower()
    if eval_split not in {"val", "test"}:
        raise ValueError(f"Training.eval_after_epoch_split must be 'val' or 'test', got: {eval_split}")

    trackers_to_eval = f"inference_{eval_split}"
    sub_out_path = f"epoch_{epoch}"
    if is_dist():
        dist.barrier()
        org_model = get_model(model)
    else:
        org_model = model

    _inference_core(
        cfg=config,
        dataset=dataset,
        model=model,
        logger=logger,
        mode=eval_split,
        sub_out_path=sub_out_path,
        require_complete_gt=eval_split == "test",
    )
    org_model.train()

    summary = {}
    eval_report = {}
    eval_time_cost = None
    if is_main_process():
        tracker_sub_folder = sub_out_path
        if eval_split == "test":
            tracker_sub_folder = os.path.join(sub_out_path, "result")
        eval_start_timestamp = time.time()
        summary, eval_report = eval_utils.evaluate(
            config,
            dataset,
            trackers_to_eval,
            tracker_sub_folder,
        )
        eval_time_cost = time.time() - eval_start_timestamp

    summary = eval_utils._broadcast_eval_summary(summary)
    eval_metric_log = None
    if summary:
        eval_metric_log = eval_utils._build_eval_metric_log(summary)
        eval_metric_log.sync()

    if is_main_process():
        _log = ""
        if eval_time_cost is not None:
            _log = f"eval time: {eval_time_cost:<6.2f} seconds\n"
        if summary:
            _log += eval_utils._format_eval_summary(summary)
        else:
            _log += "No summary output, eval failed!"
        logger.show(
            head=f"Evaluation Finish: {sub_out_path}",
            log=_log,
            write=True,
            with_header=True,
        )
        if use_tb and eval_metric_log is not None:
            eval_utils.metrics_to_tensorboard(logger.tb_eval_logger, eval_metric_log, epoch)
        eval_utils.write_default_eval_result(
            config=config,
            eval_split=eval_split,
            trackers_to_eval=trackers_to_eval,
            sub_out_path=sub_out_path,
            eval_msg=_log,
            trackeval_stdout=str(eval_report.get("stdout", "")),
            formatted_tsv=str(eval_report.get("formatted_tsv", "")),
        )
        eval_utils.copy_eval_result_to_visualize(
            config=config,
            eval_split=eval_split,
            trackers_to_eval=trackers_to_eval,
            sub_out_path=sub_out_path,
        )

    if is_dist():
        dist.barrier()
    return summary


def write_results(filename, results, dataset_name: str = ""):
    if isinstance(results, list):
        if len(results) == 0:
            results = np.empty((0, 9), dtype=np.float32)
        else:
            results = np.concatenate(results, axis=0)
    elif isinstance(results, np.ndarray):
        pass
    else:
        raise TypeError(f"Unsupported results type: {type(results)}")
    save_format = "{frame},{id},{x1:.3f},{y1:.3f},{w:.2f},{h:.2f},{s:.2f},-1,-1,-1\n"
    x_offset, y_offset = _get_submission_coordinate_offset(dataset_name=dataset_name)
    temp_filename = filename + f".tmp_{dist_rank()}"
    os.makedirs(os.path.dirname(temp_filename), exist_ok=True)
    with open(temp_filename, "w", encoding="utf-8") as f:
        for i in range(len(results)):
            frame, t_id, x1, y1, w, h, s, label, vis = results[i]
            line = save_format.format(
                frame=int(frame),
                id=int(t_id),
                x1=round(float(x1) + x_offset, 2),
                y1=round(float(y1) + y_offset, 2),
                w=round(w, 2),
                h=round(h, 2),
                s=round(s, 2),
            )
            f.write(line)
    if os.path.isfile(filename) and "inference_test" not in filename:
        os.replace(filename, f"{filename}.backup")
    os.replace(temp_filename, filename)
