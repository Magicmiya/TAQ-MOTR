import contextlib
import io
import os
import re
import shutil
import sys
from datetime import datetime
import numpy as np
import torch.distributed as dist
from rich import box
from rich.console import Console
from rich.table import Table
from torch.utils import tensorboard as tb
from data import MOTDataset
from utils import MetricLog, is_dist, is_main_process
from utils.TrackEval import trackeval

ANSI_ESCAPE_RE = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
REPORT_METRICS = ("HOTA", "CLEAR", "Identity")
SUMMARY_METRIC_ORDER = ("HOTA", "CLEAR", "Identity", "Count")
PERCENT_METRICS = {"HOTA", "CLEAR", "Identity"}
NON_PERCENT_FIELDS = {"HOTA_TP", "HOTA_FN", "HOTA_FP", "CLR_TP", "CLR_FN", "CLR_FP", "IDTP", "IDFN", "IDFP", "IDSW", "MT", "PT", "ML", "Frag", "Dets", "GT_Dets", "IDs", "GT_IDs"}
EVAL_TB_GROUPS = (
    ("eval/overview", ("HOTA", "DetA", "AssA", "MOTA", "IDF1")),
    (
        "eval/detection_localization",
        ("DetA", "DetRe", "DetPr", "CLR_Re", "CLR_Pr", "LocA"),
    ),
    (
        "eval/identity_association",
        ("AssA", "AssRe", "AssPr", "IDF1", "IDR", "IDP"),
    ),
    (
        "eval/numeric",
        ("CLR_TP", "CLR_FN", "CLR_FP", "IDSW", "Frag", "IDTP", "IDFN", "IDFP", "Dets", "GT_Dets", "IDs", "GT_IDs"),
    ),
)
EVAL_TB_OTHER_TAG = "eval/other"


def has_complete_split_gt(dataset: MOTDataset, split: str) -> bool:
    """Return whether every selected sequence has GT, rejecting partial GT splits."""
    split = str(split).strip().lower()
    selected_videos = dataset.get_selected_videos(mode=split)
    if not selected_videos:
        raise RuntimeError(f"No sequences selected for split '{split}'.")

    gt_root = os.path.join(dataset.dataset_dir, split)
    videos_with_gt = []
    videos_without_gt = []
    for video in selected_videos:
        gt_path = os.path.join(gt_root, video, "gt", "gt.txt")
        if os.path.isfile(gt_path):
            videos_with_gt.append(video)
        else:
            videos_without_gt.append(video)

    if videos_with_gt and videos_without_gt:
        raise RuntimeError(
            f"Split '{split}' contains partial GT files. "
            f"Missing gt/gt.txt for: {', '.join(videos_without_gt)}"
        )
    return bool(videos_with_gt)


class _TeeStream(io.TextIOBase):
    """Mirror writes to multiple text streams (terminal + memory buffer)."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s: str):
        for stream in self.streams:
            stream.write(s)
            stream.flush()
        return len(s)
    def flush(self):
        for stream in self.streams:
            stream.flush()
    def isatty(self) -> bool:
        first = self.streams[0] if self.streams else None
        return bool(getattr(first, "isatty", lambda: False)())
    @property
    def encoding(self):
        first = self.streams[0] if self.streams else None
        return getattr(first, "encoding", "utf-8")

def metrics_to_tensorboard(writer: tb.writer.SummaryWriter | None, metrics: MetricLog, epoch: int):
    """Write epoch metrics into tensorboard grouped scalar tags."""
    if writer is None:
        return
    tags, tag_scalar_dicts = metrics.get_tb(mode="epochs")
    for tag, scalar_dict in zip(tags, tag_scalar_dicts):
        writer.add_scalars(main_tag=tag, tag_scalar_dict=scalar_dict, global_step=epoch)

def _get_tracker_eval_dir(config: dict, eval_split: str, trackers_to_eval: str, sub_out_path: str) -> str:
    """Return the checkpoint-specific evaluation directory under inference outputs."""
    split = str(eval_split).lower()
    # Use the actual evaluated split instead of config["MODE"], so online train-time val reports
    # land beside the generated tracker txt files.
    return os.path.join(config["OUTPUTS_DIR"], split, trackers_to_eval, sub_out_path)

def _get_visualize_dir(config: dict, sub_out_path: str) -> str:
    """Return the visualize directory dedicated to one evaluated result."""
    # Suffix visualize directories with the evaluated result name to avoid cross-run overwrites.
    return os.path.join(config["OUTPUTS_DIR"], f"visualize_{sub_out_path}")

def _format_metric_tables_plain_text(metric_tables: list[dict]) -> str:
    """Render full metric tables for file export without Rich truncation."""
    sections = []
    for table in metric_tables:
        headers = [str(x) for x in table.get("headers", [])]
        rows = [[str(x) for x in row] for row in table.get("rows", [])]
        if not headers:
            continue

        widths = [len(h) for h in headers]
        for row in rows:
            for idx, cell in enumerate(row):
                widths[idx] = max(widths[idx], len(cell))

        def _render_row(row: list[str]) -> str:
            rendered = []
            for idx, cell in enumerate(row):
                rendered.append(cell.ljust(widths[idx]) if idx == 0 else cell.rjust(widths[idx]))
            return "  ".join(rendered)

        lines = [str(table.get("title", "")).strip(), "", _render_row(headers)]
        lines.append("  ".join("-" * width for width in widths))
        for row in rows:
            lines.append(_render_row(row))
        sections.append("\n".join(lines).rstrip())
    return "\n\n".join(section for section in sections if section)

def evaluate(cfg, dataset: MOTDataset, trackers_to_eval: str, tracker_sub_folder: str):
    """Run TrackEval and return summary/report while preserving current side effects."""
    if not is_main_process():
        return {}, {}

    def _pick_eval_class(tracker_res: dict, allow_cls_comb: bool = False) -> str | None:
        combined = tracker_res.get("COMBINED_SEQ", {})
        if "pedestrian" in combined:
            return "pedestrian"
        for key in combined.keys():
            key_str = str(key)
            if allow_cls_comb or not key_str.startswith("cls_comb"):
                return key_str
        return None

    def _compact_number(value) -> str:
        if isinstance(value, (np.integer, int)):
            return str(int(value))
        if isinstance(value, (np.floating, float)):
            f_val = float(value)
            if not np.isfinite(f_val):
                return str(f_val)
            if abs(f_val - round(f_val)) < 1e-9:
                return str(int(round(f_val)))
            return f"{f_val:.2f}".rstrip("0").rstrip(".")
        return str(value)

    split = str(dataset.mode).lower()
    # Evaluation artifacts are split-scoped; tracker txt, TrackEval temp files,
    # and readable reports all share OUTPUTS_DIR/<split>/<tracker>/<run>.
    out_root = os.path.join(cfg["OUTPUTS_DIR"], split)
    data_dir, dataset_name = dataset.dataset_dir, dataset.name
    selected_videos = dataset.get_selected_videos(mode=split)
    if dataset_name in {"DanceTrack", "SportsMOT", "BFT"}:
        num_cores, display_less_progress, benchmark = 16, False, "MOT17"
    elif dataset_name == "MOT17":
        num_cores, display_less_progress, benchmark = 8, True, "MOT17"
    elif dataset_name == "MOT20":
        num_cores, display_less_progress, benchmark = 8, True, "MOT20"
    else:
        raise NotImplementedError(f"Do not support this Dataset name: {dataset_name}")

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update({"USE_PARALLEL": True, "NUM_PARALLEL_CORES": num_cores, "PRINT_RESULTS": False, "PRINT_ONLY_COMBINED": False, "PRINT_CONFIG": False, "TIME_PROGRESS": False, "DISPLAY_LESS_PROGRESS": display_less_progress, "OUTPUT_SUMMARY": True, "OUTPUT_DETAILED": True, "PLOT_CURVES": False})
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_overrides = {}
    if hasattr(dataset, "get_trackeval_dataset_config"):
        dataset_overrides = dataset.get_trackeval_dataset_config(mode=split)
    dataset_config.update({
        "GT_FOLDER": os.path.join(data_dir, split),
        "TRACKERS_FOLDER": out_root,
        "TRACKERS_TO_EVAL": [trackers_to_eval],
        "TRACKER_SUB_FOLDER": tracker_sub_folder,
        "SEQ_INFO": {video: None for video in selected_videos},
        "SEQMAP_FILE": None,
        "SKIP_SPLIT_FOL": True,
        "SPLIT_TO_EVAL": split,
        "BENCHMARK": benchmark,
        "PRINT_CONFIG": False,
    })
    dataset_config.update(dataset_overrides)
    metrics_config = {"METRICS": list(REPORT_METRICS), "THRESHOLD": 0.5, "PRINT_CONFIG": False}
    selected_metrics = [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]
    metrics_list = [m(metrics_config) for m in selected_metrics if m.get_name() in metrics_config["METRICS"]]
    if not metrics_list:
        raise RuntimeError("No TrackEval metrics selected.")
    report_metrics = metrics_list + [trackeval.metrics.Count()]
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]

    stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
    tee_stdout, tee_stderr = _TeeStream(sys.stdout, stdout_buf), _TeeStream(sys.stderr, stderr_buf)
    with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(tee_stderr):
        output_res, output_msg = evaluator.evaluate(dataset_list, metrics_list)
        dataset_key = next(iter(output_res.keys()), dataset_name)
        tracker_res = output_res.get(dataset_key, {}).get(trackers_to_eval, {})
        cls_name = _pick_eval_class(tracker_res) if isinstance(tracker_res, dict) else None
        metric_tables = []
        if cls_name is not None:
            seq_keys = sorted([k for k in tracker_res.keys() if k != "COMBINED_SEQ"])
            if "COMBINED_SEQ" in tracker_res:
                seq_keys.append("COMBINED_SEQ")
            for metric in report_metrics:
                metric_name = metric.get_name()
                if metric_name not in tracker_res.get("COMBINED_SEQ", {}).get(cls_name, {}):
                    continue
                fields = list(getattr(metric, "summary_fields", []))
                if not fields:
                    continue
                rows = []
                for seq in seq_keys:
                    metric_data = tracker_res.get(seq, {}).get(cls_name, {}).get(metric_name, None)
                    if not isinstance(metric_data, dict):
                        continue
                    values = []
                    for field in fields:
                        if field not in metric_data:
                            values.append("")
                            continue
                        value = metric_data[field]
                        if field in getattr(metric, "float_array_fields", []):
                            value = 100.0 * float(np.mean(value))
                        elif field in getattr(metric, "float_fields", []):
                            value = 100.0 * float(value)
                        elif field in getattr(metric, "integer_fields", []):
                            value = int(value)
                        values.append(_compact_number(value))
                    rows.append(["COMBINED" if seq == "COMBINED_SEQ" else str(seq)] + values)
                if rows:
                    metric_tables.append({"title": f"{metric_name}: {trackers_to_eval}-{cls_name}", "headers": ["seq"] + fields, "rows": rows})

        if metric_tables:
            console = Console(file=tee_stdout, color_system="auto", width=1000)
            for idx, table_data in enumerate(metric_tables):
                table = Table(title=table_data["title"], box=box.SIMPLE_HEAVY, header_style="bold cyan", title_style="bold green", show_lines=False)
                for col_idx, header in enumerate(table_data["headers"]):
                    table.add_column(str(header), justify="left" if col_idx == 0 else "right", no_wrap=True)
                for row in table_data["rows"]:
                    row_cells = [str(x) for x in row]
                    table.add_row(*row_cells, style=("bold yellow" if row_cells and row_cells[0] == "COMBINED" else None))
                console.print(table)
                if idx < len(metric_tables) - 1:
                    console.print()

    plain_text_tables = _format_metric_tables_plain_text(metric_tables)
    trackeval_stdout = stdout_buf.getvalue()
    err_text = stderr_buf.getvalue()
    if err_text.strip():
        trackeval_stdout = f"{trackeval_stdout.rstrip()}\n{err_text.rstrip()}\n"

    # Persist readable artifacts in the checkpoint-specific sub-folder only.
    dst_dir = _get_tracker_eval_dir(cfg, split, trackers_to_eval, tracker_sub_folder)
    os.makedirs(dst_dir, exist_ok=True)
    src_summary, src_detail = os.path.join(out_root, trackers_to_eval, "pedestrian_summary.txt"), os.path.join(out_root, trackers_to_eval, "pedestrian_detailed.csv")
    summary = {}
    if os.path.isfile(src_summary):
        with open(src_summary) as f:
            metric_names = f.readline()[:-1].split(" ")
            metric_values = f.readline()[:-1].split(" ")
        summary = {n: float(v) for n, v in zip(metric_names, metric_values)}
        os.remove(src_summary)
    if os.path.isfile(src_detail):
        os.remove(src_detail)

    if not summary:
        tracker_res = output_res.get(dataset_key, {}).get(trackers_to_eval, {})
        cls_name = _pick_eval_class(tracker_res, allow_cls_comb=True) if isinstance(tracker_res, dict) else None
        if cls_name is not None:
            combined = tracker_res.get("COMBINED_SEQ", {}).get(cls_name, {})
            for metric_name in SUMMARY_METRIC_ORDER:
                metric_data = combined.get(metric_name, None)
                if not isinstance(metric_data, dict):
                    continue
                for field_name, value in metric_data.items():
                    if isinstance(value, np.ndarray):
                        if value.size == 0:
                            continue
                        field_value = float(np.mean(value))
                    elif isinstance(value, (np.floating, float, np.integer, int)):
                        field_value = float(value)
                    else:
                        continue
                    if metric_name in PERCENT_METRICS and field_name not in NON_PERCENT_FIELDS:
                        field_value *= 100.0
                    summary[field_name] = field_value

    def _cell(val: str, idx: int) -> str:
        s = str(val)
        min_width = 12 if idx == 0 else 4
        return s.ljust(min_width) if len(s) < min_width else s

    sections = []
    for table in metric_tables:
        lines = [str(table["title"])]
        headers = [str(x) for x in table["headers"]]
        lines.append("\t".join(_cell(h, i) for i, h in enumerate(headers)))
        for row in table["rows"]:
            row_cells = [str(x) for x in row]
            lines.append("\t".join(_cell(v, i) for i, v in enumerate(row_cells)))
        sections.append("\n".join(lines))
    if summary:
        lines = ["SUMMARY: overall", "\t".join([_cell("metric", 0), _cell("value", 1)])]
        for key, value in summary.items():
            lines.append("\t".join([_cell(str(key), 0), _cell(_compact_number(value), 1)]))
        sections.append("\n".join(lines))
    formatted_tsv = "\n\n".join(sections)

    export_stdout = plain_text_tables if plain_text_tables.strip() else trackeval_stdout
    report = {"stdout": export_stdout, "formatted_tsv": formatted_tsv, "output_msg": output_msg}
    return summary, report


def _sanitize_console_text_for_file(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    cleaned = ANSI_ESCAPE_RE.sub("", text)
    out_chars = []
    for ch in cleaned:
        if ch != "\r":
            out_chars.append("-" if 0x2500 <= ord(ch) <= 0x257F else ch)
    return "".join(out_chars)

def _write_eval_result_files(out_dir: str, sub_out_path: str, eval_msg: str, trackeval_stdout: str = "", formatted_tsv: str = ""):
    """Write human-readable evaluation artifacts to a target directory."""
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "eval_result.txt")
    out_tsv = os.path.join(out_dir, "eval_result.tsv")

    # Store one self-contained readable report per evaluated result instead of TrackEval's raw summary/detail pair.
    finish_text = "\n".join(["=" * 80, f"Evaluation Finish: {sub_out_path}", "=" * 80, eval_msg.rstrip()])
    block = []
    safe_stdout = _sanitize_console_text_for_file(trackeval_stdout)
    if safe_stdout.strip():
        block.append(safe_stdout.rstrip())
    block.append(finish_text)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(block).rstrip())
        f.write("\n")

    if formatted_tsv.strip():
        with open(out_tsv, "w", encoding="utf-8") as f:
            f.write(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  {sub_out_path}\n")
            f.write(formatted_tsv.rstrip())
            f.write("\n")

def _format_eval_summary(summary: dict, metrics_per_line: int = 8) -> str:
    if not summary:
        return ""
    items, max_len, lines = list(summary.items()), max(len(k) for k in summary.keys()), []
    for i in range(0, len(items), metrics_per_line):
        cur = items[i : i + metrics_per_line]
        lines += [" ".join(f"{k:<{max_len}}" for k, _ in cur), " ".join(f"{v:<{max_len}}" for _, v in cur)]
        if i + metrics_per_line < len(items):
            lines.append("")
    return "\n".join(lines)


def _get_eval_result_archive_dirs(config: dict) -> list[str]:
    """Return stable result archive directories for the current eval dataset."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_root = os.path.join(repo_root, "outputs", "results")
    archive_dirs = [results_root]

    dataset_cfg = config.get("Dataset", {})
    dataset_name = str(dataset_cfg.get("dataset_name", "")).strip()
    dataset_name = re.sub(r"[\\/]+", "_", dataset_name).strip(" .")
    if dataset_name:
        # Mirror latest eval text into dataset-specific folders so cross-dataset comparisons stay separated.
        archive_dirs.append(os.path.join(results_root, dataset_name))
    return archive_dirs


def write_default_eval_result(
    config: dict,
    eval_split: str,
    trackers_to_eval: str,
    sub_out_path: str,
    eval_msg: str,
    trackeval_stdout: str = "",
    formatted_tsv: str = "",
):
    """Write the default readable evaluation report under inference outputs."""
    if not is_main_process():
        return
    if (not trackeval_stdout.strip()) and (not eval_msg.strip()) and (not formatted_tsv.strip()):
        return

    out_dir = _get_tracker_eval_dir(config, eval_split, trackers_to_eval, sub_out_path)
    _write_eval_result_files(
        out_dir=out_dir,
        sub_out_path=sub_out_path,
        eval_msg=eval_msg,
        trackeval_stdout=trackeval_stdout,
        formatted_tsv=formatted_tsv,
    )

    # Keep the historical root-level latest pointer while also archiving by dataset name.
    src_eval_result = os.path.join(out_dir, "eval_result.txt")
    for archive_dir in _get_eval_result_archive_dirs(config):
        os.makedirs(archive_dir, exist_ok=True)
        shutil.copyfile(src_eval_result, os.path.join(archive_dir, "last_eval_result.txt"))

def copy_eval_result_to_visualize(config: dict, eval_split: str, trackers_to_eval: str, sub_out_path: str):
    """Copy readable evaluation reports into the visualize directory for the same result."""
    if not is_main_process() or not bool(config.get("visualize", False)):
        return

    src_dir = _get_tracker_eval_dir(config, eval_split, trackers_to_eval, sub_out_path)
    dst_dir = _get_visualize_dir(config, sub_out_path)
    # Keep visualize text outputs identical to inference_val outputs by copying the generated canonical files.
    os.makedirs(dst_dir, exist_ok=True)
    for file_name in ("eval_result.txt", "eval_result.tsv"):
        src_path = os.path.join(src_dir, file_name)
        if os.path.isfile(src_path):
            shutil.copyfile(src_path, os.path.join(dst_dir, file_name))

def _get_eval_step(sub_out_path: str) -> int | None:
    """Parse eval step index from names like epoch_12."""
    try:
        return int(sub_out_path.split("_")[-1])
    except (ValueError, IndexError):
        return None

def _broadcast_eval_summary(summary: dict | None) -> dict:
    """Broadcast main-process summary to all ranks in distributed mode."""
    if not is_dist():
        return summary or {}
    obj_list = [summary]
    dist.broadcast_object_list(obj_list, src=0)
    return obj_list[0] or {}

def _add_eval_tb_group(metric_log: MetricLog, tag: str, summary: dict, names: tuple[str, ...]) -> set[str]:
    """Add one TensorBoard scalar group and return metrics written to it."""
    written = set()
    for name in names:
        if name not in summary:
            continue
        metric_log.add_scalar(tag, name, summary[name])
        written.add(name)
    return written

def _build_eval_metric_log(summary: dict) -> MetricLog:
    """Convert flat TrackEval summary into task-oriented TensorBoard groups."""
    metric_log = MetricLog()
    if not summary:
        return metric_log

    written = set()
    for tag, names in EVAL_TB_GROUPS:
        written.update(_add_eval_tb_group(metric_log, tag, summary, names))

    # Keep explicitly grouped metrics focused, then preserve every remaining
    # TrackEval scalar in a single fallback card for auditability.
    other_names = tuple(name for name in summary.keys() if name not in written)
    _add_eval_tb_group(metric_log, EVAL_TB_OTHER_TAG, summary, other_names)
    return metric_log
