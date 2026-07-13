import copy
import itertools
import os, subprocess
import argparse
import re
import yaml
import torch.distributed as dist
from datetime import datetime

SWEEP_NAME_ALIASES = {
    "Decoder.hqg_num_learnable_memory": "LM",
    "Decoder.hqg_active_learnable_memory": "ALM",
    "Decoder.hqg_num_blocks": "HB",
    "Decoder.qpn_interact_max_state": "QIS",
    "Decoder.num_queries": "NQ",
    "Life_cycle_management.new_born_threshold": "NB",
    "Life_cycle_management.track_thresh": "TT",
    "Life_cycle_management.high_conf_threshold": "HC",
    "Life_cycle_management.Sudden_death_threshold": "SD",
    "Inference.result_score_thresh": "RS",
    "Inference.result_min_area": "RMA",
}

SWEEP_DISALLOWED_KEYS = {
    "EXP_NAME",
    "CONFIG_PATH",
    "MODE",
    "OUTPUTS_DIR",
    "NEW_OUT_DIR",
    "Training.Available_gpus",
    "Training.use_distributed",
    "Dataset.dataset_root",
}

REQUIRED_EVAL_MODEL_KEYS = (
    "MODULE_NAME",
    "Backbone",
    "Encoder",
    "Decoder",
    "Criterion",
    "Query_updater",
    "Life_cycle_management",
    "hidden_dim",
    "num_classes",
    "batch_norm",
)

STRICT_TRAIN_MODEL_KEYS = (
    "MODULE_NAME",
    "Backbone",
    "Encoder",
    "Decoder",
    "Criterion",
    "Query_updater",
    "hidden_dim",
    "num_classes",
    "batch_norm",
)

SECTION_STRICT_TRAIN_KEYS = {
    "Life_cycle_management": (
        "hidden_dim",
        "num_classes",
    ),
}

SECTION_EVAL_OVERRIDE_KEYS = {
    "Decoder": (
        "num_queries",
        "hqg_init_feat_level",
        "hqg_init_feat_levels",
        "qpn_interact_max_state",
        "hqg_num_learnable_memory",
        "hqg_active_learnable_memory",
        "hqg_num_blocks",
        "hqg_det_dn_mask_track_memory",
    ),
}


def parse_option():
    # =================== Exp Configration ===================
    parser = argparse.ArgumentParser("The main script of TAQ-MOTR", add_help=True)
    parser.add_argument("-E", "--exp_name", type=str, help="Experiment name", required=True)
    parser.add_argument("-C", "--config_path", type=str, help="Config files path", required=True)
    parser.add_argument("-M", "--mode", type=str, help="script mode config such as train,is_eval,submit", default=None)
    parser.add_argument("-O", "--outputs_dir", type=str, help="Output directory", default=None)
    parser.add_argument("-NEW", "--new_out_dir", type=bool, help="Use time as the suffix of the output directory")
    parser.add_argument("-P", "--pretrained_model", type=str, help="Pretrained model file path", default=None)

    parser.add_argument(
        "-L", "--checkpoint_level", type=str, help="Level of checkpoint training,-1 for auto", default=None
    )
    parser.add_argument("-R", "--resume", type=str, help="The file path for the resume parameter", default=None)
    parser.add_argument(
        "-V", "--videos", type=str, help="Comma-separated eval videos, e.g. dancetrack0041,dancetrack0043", default=None
    )
    parser.add_argument("--gpus", type=str, help="Comma-separated GPU ids, e.g. 0,1,2,3", default=None)
    parser.add_argument("--distributed", action="store_true", default=None, help="Enable distributed train/eval runtime")
    parser.add_argument("-EFP", "--eval_file_path", type=str, help="Checkpoint file or directory for evaluation", default=None)
    parser.add_argument("-DR", "--dataset_root", type=str, help="Dataset root directory", default=None)
    parser.add_argument(
        "--sweep",
        nargs="*",
        default=None,
        help="Ablation overrides, e.g. Decoder.hqg_num_learnable_memory=16,32",
    )
    return parser.parse_args()


def merge_configs(config: dict, option: argparse.Namespace) -> dict:
    # Merge parser option and .yaml config,
    for option_k, option_v in vars(option).items():
        if option_k == "sweep":
            continue
        if option_k == "distributed":
            if option_v is not None:
                config["Training"]["use_distributed"] = True
            continue
        if option_v is not None:
            if option_k.upper() in config:
                config[option_k.upper()] = option_v
            elif option_k.lower() == "gpus" and isinstance(option_v, str):
                # Keep runtime GPU selection configurable without editing per-dataset YAML files.
                config["Training"]["Available_gpus"] = option_v
            elif option_k.lower() == "dataset_root" and isinstance(option_v, str):
                config["Dataset"]["dataset_root"] = option_v
            elif option_k.lower() == "videos" and isinstance(option_v, str):
                config["Dataset"]["eval_videos"] = [video.strip() for video in option_v.split(",") if video.strip()]
            else:
                raise KeyError(f"Option '{option_k}' not found in config file")
    return config


def _config_base_dir() -> str:
    return os.path.dirname(__file__)


def _resolve_single_config_path(config_path: str) -> str:
    if os.path.isabs(config_path):
        return config_path
    config_dir_path = os.path.normpath(os.path.join(_config_base_dir(), config_path))
    cwd_path = os.path.abspath(config_path)
    if os.path.exists(config_dir_path):
        return config_dir_path
    if os.path.exists(cwd_path):
        return cwd_path
    return config_dir_path


def _resolve_config_paths(config_path: str) -> list[str]:
    raw_paths = [path.strip() for path in str(config_path).split(",") if path.strip()]
    if not raw_paths:
        raise ValueError("Config path is empty.")

    resolved_paths: list[str] = []
    for raw_path in raw_paths:
        resolved_path = _resolve_single_config_path(raw_path)
        if os.path.isdir(resolved_path):
            yaml_paths = [
                os.path.join(resolved_path, name)
                for name in sorted(os.listdir(resolved_path))
                if name.endswith((".yaml", ".yml"))
            ]
            if not yaml_paths:
                raise FileNotFoundError(f"No yaml config files found in directory: {resolved_path}")
            resolved_paths.extend(yaml_paths)
        elif os.path.isfile(resolved_path):
            resolved_paths.append(resolved_path)
        else:
            raise FileNotFoundError(f"Config file or directory not found at: {resolved_path}")
    return resolved_paths


def _load_yaml_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f.read())
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid yaml config format at: {config_path}")
    return cfg


def _option_for_config(option: argparse.Namespace, config_path: str) -> argparse.Namespace:
    option_copy = argparse.Namespace(**vars(option))
    option_copy.config_path = config_path
    return option_copy


def _apply_runtime_config_lock(config: dict, available_gpus: str, use_distributed: bool):
    config["Training"]["Available_gpus"] = available_gpus
    config["Training"]["use_distributed"] = use_distributed


def _runtime_lock_from_first_config(first_config: dict, option: argparse.Namespace) -> tuple[str, bool]:
    training_cfg = first_config.get("Training", {})
    if not isinstance(training_cfg, dict):
        raise KeyError("Missing Training section in the first config.")

    available_gpus = option.gpus if option.gpus is not None else training_cfg.get("Available_gpus", None)
    if available_gpus is None:
        raise KeyError("Missing Training.Available_gpus in the first config and --gpus was not provided.")

    use_distributed = True if option.distributed is not None else bool(training_cfg.get("use_distributed", False))
    return str(available_gpus), bool(use_distributed)


def _parse_sweep_value(raw_value: str):
    return yaml.safe_load(raw_value)


def _parse_sweep_specs(sweep_specs: list[str] | None) -> list[tuple[dict[str, object], str]]:
    if not sweep_specs:
        return [({}, "")]

    parsed_specs: list[tuple[str, list[object]]] = []
    for spec in sweep_specs:
        if "=" not in spec:
            raise ValueError(f"Invalid sweep spec '{spec}', expected key=v1,v2.")
        key, raw_values = spec.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid sweep spec '{spec}', key is empty.")
        if key in SWEEP_DISALLOWED_KEYS:
            raise ValueError(f"Sweep key '{key}' is not allowed in one runtime execution.")

        values = [_parse_sweep_value(value.strip()) for value in raw_values.split(",") if value.strip()]
        if not values:
            raise ValueError(f"Invalid sweep spec '{spec}', no values found.")
        parsed_specs.append((key, values))

    expanded: list[tuple[dict[str, object], str]] = []
    keys = [key for key, _ in parsed_specs]
    value_lists = [values for _, values in parsed_specs]
    for values in itertools.product(*value_lists):
        overrides = dict(zip(keys, values))
        suffix = "_".join(_build_sweep_label(key=key, value=value) for key, value in overrides.items())
        expanded.append((overrides, suffix))
    return expanded


def _set_by_path(config: dict, key_path: str, value):
    parts = [part for part in str(key_path).split(".") if part]
    if not parts:
        raise ValueError("Empty config override path.")

    cursor = config
    for part in parts[:-1]:
        if isinstance(cursor, list):
            cursor = cursor[int(part)]
        else:
            if part not in cursor:
                cursor[part] = {}
            cursor = cursor[part]

    final_key = parts[-1]
    if isinstance(cursor, list):
        cursor[int(final_key)] = value
    else:
        cursor[final_key] = value


def _abbrev_from_key(key: str) -> str:
    leaf = str(key).split(".")[-1]
    return "".join(word[:1].upper() for word in leaf.split("_") if word)


def _sanitize_name_part(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9.\-]+", "_", str(value).strip())
    return sanitized.strip("_")


def _format_sweep_value(value) -> str:
    if isinstance(value, bool):
        return "T" if value else "F"
    if value is None:
        return "None"
    return _sanitize_name_part(str(value))


def _build_sweep_label(key: str, value) -> str:
    prefix = SWEEP_NAME_ALIASES.get(key, _abbrev_from_key(key))
    return f"{prefix}{_format_sweep_value(value)}"


def _build_exp_name(base_exp_name: str, config_stem: str, sweep_suffix: str, use_config_suffix: bool) -> str:
    name_parts = [base_exp_name]
    if use_config_suffix:
        name_parts.append(config_stem)
    if sweep_suffix:
        name_parts.append(sweep_suffix)
    return "_".join(_sanitize_name_part(part) for part in name_parts if str(part).strip())


def get_git_version() -> str:
    try:
        git_version = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
        return git_version
    except subprocess.CalledProcessError:
        return "unknown"


def _resolve_eval_root(eval_file_path: str) -> str:
    # Keep eval root resolving in one place so both config-merge and output-dir logic use identical rules.
    if os.path.isfile(eval_file_path) and eval_file_path.endswith(".pth"):
        return os.path.dirname(eval_file_path)
    if os.path.isdir(eval_file_path):
        return eval_file_path
    raise FileNotFoundError(f"{eval_file_path} must be a *.pth file or a directory path")


def _build_eval_output_path(config: dict) -> str:
    eval_file_path = str(config["EVAL_FILE_PATH"])
    eval_root = _resolve_eval_root(eval_file_path)
    # Keep eval OUTPUTS_DIR pinned to the checkpoint root so val/test artifacts
    # always land under <checkpoint_root>/<split>/inference_<split>/<run_name>.
    return eval_root


def _try_load_checkpoint_train_config(eval_root: str) -> dict | None:
    train_config_path = os.path.join(eval_root, "train", "config.yaml")
    if not os.path.isfile(train_config_path):
        print(f"[Warining] train config not found at: {train_config_path}. Fallback to eval startup config.")
        return None

    try:
        with open(train_config_path) as f:
            train_cfg = yaml.load(f.read(), yaml.FullLoader)
    except Exception as e:
        print(
            f"[Warining] failed to read train config at: {train_config_path}, reason: {e}. "
            f"Fallback to eval startup config."
        )
        return None

    if not isinstance(train_cfg, dict):
        print(f"[Warining] invalid train config format at: {train_config_path}. Fallback to eval startup config.")
        return None
    return train_cfg


def _missing_model_config_keys(cfg: dict) -> list[str]:
    return [key for key in REQUIRED_EVAL_MODEL_KEYS if key not in cfg]


def _deep_merge_dict(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _is_rank_zero_before_dist_init() -> bool:
    # Config loading happens before init_process_group, so use torchrun's env rank for log filtering.
    return os.environ.get("RANK", "0") == "0"


def _merge_eval_with_train_model_config(eval_cfg: dict, train_cfg: dict, train_cfg_path: str):
    # Keep checkpoint-bound model structure fixed while letting eval yaml override runtime inference knobs.
    eval_override_sections = {
        section: dict(eval_cfg.get(section, {}))
        for section in SECTION_EVAL_OVERRIDE_KEYS
        if isinstance(eval_cfg.get(section), dict)
    }

    for key in STRICT_TRAIN_MODEL_KEYS:
        if key in train_cfg:
            if isinstance(eval_cfg.get(key), dict) and isinstance(train_cfg[key], dict):
                eval_cfg[key] = _deep_merge_dict(eval_cfg[key], train_cfg[key])
            else:
                eval_cfg[key] = train_cfg[key]

    for section, strict_keys in SECTION_STRICT_TRAIN_KEYS.items():
        train_section = train_cfg.get(section)
        eval_section = eval_cfg.get(section)
        if not isinstance(train_section, dict):
            continue

        if isinstance(eval_section, dict):
            merged_section = _deep_merge_dict(train_section, eval_section)
        else:
            merged_section = dict(train_section)

        for key in strict_keys:
            if key in train_section:
                merged_section[key] = train_section[key]
        eval_cfg[section] = merged_section

    for section, override_keys in SECTION_EVAL_OVERRIDE_KEYS.items():
        eval_section = eval_override_sections.get(section)
        merged_section = eval_cfg.get(section)
        if not isinstance(eval_section, dict) or not isinstance(merged_section, dict):
            continue

        for key in override_keys:
            if key in eval_section:
                # Allow selected eval-time decoder knobs such as num_queries to override checkpoint train config.
                merged_section[key] = eval_section[key]

    if _is_rank_zero_before_dist_init():
        print(f"[Info] use model config from checkpoint train config: {train_cfg_path}")


def _resolve_outputs_root(config: dict, project_root: str) -> str:
    if os.path.isabs(config["OUTPUTS_DIR"]):
        return config["OUTPUTS_DIR"]
    return os.path.join(project_root, config["OUTPUTS_DIR"])


def _build_output_path(config: dict, outputs_root: str, defer_timestamp: bool = False, apply_side_effects: bool = True) -> str:
    mode = str(config["MODE"]).lower()
    output_path = ""

    if config["RESUME"] is not None and config["MODE"] == "train":
        if os.path.isfile(config["RESUME"]):
            if config["RESUME"].startswith(outputs_root):
                # when resume file path is in the outputs_root directly use this file dir as output_path
                output_path = os.path.dirname(config["RESUME"])
            else:
                if defer_timestamp:
                    output_path = os.path.join(outputs_root, config["EXP_NAME"])
                else:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    org_out_path = os.path.dirname(config["RESUME"])
                    output_path = str(os.path.join(outputs_root, config["EXP_NAME"], f"_resume_{timestamp}"))
                    if apply_side_effects:
                        os.makedirs(output_path, exist_ok=True)
                        os.system(f"cp -r {os.path.join(org_out_path, 'train')} {os.path.join(output_path, 'train')}")
        else:
            raise FileNotFoundError(f"resume file not found at: {config['RESUME']}, it should be absolute path ")
    elif mode in ["val", "test"]:
        output_path = _build_eval_output_path(config)
        if apply_side_effects:
            os.makedirs(output_path, exist_ok=True)
    else:
        output_path = os.path.join(outputs_root, config["EXP_NAME"])
        if config.get("NEW_OUT_DIR", True):
            if not defer_timestamp:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"{output_path}_{timestamp}"
                if apply_side_effects:
                    os.makedirs(output_path, exist_ok=True)
        else:
            if apply_side_effects:
                # Only clear the train directory to avoid deleting important .pth files
                train_dir = os.path.join(str(output_path), "train")
                if os.path.exists(train_dir):
                    os.system(f"rm -r {train_dir}")

    return output_path


def sync_distributed_train_output_dir(config: dict) -> dict:
    if str(config.get("MODE", "")).lower() != "train":
        return config
    if not bool(config.get("Training", {}).get("use_distributed", False)):
        return config
    if not (dist.is_available() and dist.is_initialized()):
        return config

    outputs_root = config.get("_OUTPUTS_ROOT", None)
    if outputs_root is None:
        raise KeyError("Missing _OUTPUTS_ROOT in config during distributed output-dir sync.")

    shared_output_path = None
    if dist.get_rank() == 0:
        shared_output_path = _build_output_path(
            config=config,
            outputs_root=str(outputs_root),
            defer_timestamp=False,
            apply_side_effects=True,
        )

    object_list = [shared_output_path]
    dist.broadcast_object_list(object_list, src=0)
    config["OUTPUTS_DIR"] = str(object_list[0])
    # All ranks must agree on one run directory, otherwise validation txt files split across sibling folders.
    os.makedirs(config["OUTPUTS_DIR"], exist_ok=True)
    return config


def _finalize_config(config: dict) -> dict:
    mode = str(config["MODE"]).lower()
    if mode in ["val", "test"]:
        eval_root = _resolve_eval_root(config["EVAL_FILE_PATH"])
        train_cfg_path = os.path.join(eval_root, "train", "config.yaml")
        train_cfg = _try_load_checkpoint_train_config(eval_root)
        if train_cfg is not None:
            _merge_eval_with_train_model_config(config, train_cfg, train_cfg_path)
        else:
            # Runtime-only eval yamls rely on checkpoint-side train/config.yaml for model construction keys.
            missing_keys = _missing_model_config_keys(config)
            if missing_keys:
                raise KeyError(
                    f"Missing eval model config keys: {missing_keys}. "
                    f"Either restore checkpoint train config at {train_cfg_path} or add these keys back to eval yaml."
                )

    config["GIT_VERSION"] = get_git_version()

    project_root = os.path.abspath(os.path.join(__file__, "../.."))  # Get project root path
    outputs_root = _resolve_outputs_root(config, project_root)
    config["_OUTPUTS_ROOT"] = outputs_root

    defer_timestamp = bool(mode == "train" and config.get("Training", {}).get("use_distributed", False))
    output_path = _build_output_path(
        config=config,
        outputs_root=outputs_root,
        defer_timestamp=defer_timestamp,
        apply_side_effects=not defer_timestamp,
    )

    config["OUTPUTS_DIR"] = output_path
    return config


def get_config() -> list[dict]:
    """get runtime configration and merge configs"""
    opt = parse_option()
    config_paths = _resolve_config_paths(opt.config_path)
    loaded_configs = [_load_yaml_config(config_path) for config_path in config_paths]
    available_gpus, use_distributed = _runtime_lock_from_first_config(loaded_configs[0], opt)
    sweep_items = _parse_sweep_specs(opt.sweep)
    use_config_suffix = len(config_paths) > 1

    config_list: list[dict] = []
    for config_path, loaded_config in zip(config_paths, loaded_configs):
        config_stem = os.path.splitext(os.path.basename(config_path))[0]
        merged_config = merge_configs(copy.deepcopy(loaded_config), _option_for_config(opt, config_path))
        _apply_runtime_config_lock(
            config=merged_config,
            available_gpus=available_gpus,
            use_distributed=use_distributed,
        )

        for sweep_overrides, sweep_suffix in sweep_items:
            sweep_config = copy.deepcopy(merged_config)
            for key, value in sweep_overrides.items():
                _set_by_path(sweep_config, key, value)
            sweep_config["EXP_NAME"] = _build_exp_name(
                base_exp_name=opt.exp_name,
                config_stem=config_stem,
                sweep_suffix=sweep_suffix,
                use_config_suffix=use_config_suffix,
            )
            config_list.append(_finalize_config(sweep_config))

    return config_list
