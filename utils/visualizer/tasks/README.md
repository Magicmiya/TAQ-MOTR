# Visualizer Tasks README

This document describes the visual tasks under `utils/visualizer/tasks`, including their purpose, lifecycle, input interface, configuration keys, outputs, and extension points. It is intended as a review reference for the current task set.

## 1. Overall structure

The visualization system consists of three layers:

1. `Visualizer` in [core.py](../core.py)
   - Builds task instances from config.
   - Collects hook events from `TensorHook`.
   - Collects runtime statistics from `GetTime`.
   - Dispatches frame updates to enabled tasks.
2. `BaseVisualTask`
   - Shared task lifecycle interface.
   - Each task decides what hook switches it needs and what files it exports.
3. Concrete tasks under this directory
   - Each task focuses on one independent visualization or summary job.

Current built-in tasks are registered in:
- [__init__.py](__init__.py)
- [core.py](../core.py)

Current runtime model:
- `TensorHook` stores CPU snapshots only.
- `Visualizer` dispatches frame jobs to a shared frame-worker pool.
- Per-frame image writes are offloaded to a shared writer pool.
- Video-level summaries are aggregated by `video_name` and flushed in `close()`.

## 2. Common interfaces

### 2.1 FrameContext

Defined in [core.py](../core.py).

```python
FrameContext(
    mode: str,
    video_name: str,
    frame_id: int,
    img_path: str,
    image_bgr: Optional[np.ndarray],
    track_result: np.ndarray,
    tag: Optional[str],
)
```

Main fields:
- `mode`: running mode, e.g. `train` or `val`
- `video_name`: current video id
- `frame_id`: current frame index
- `img_path`: image path on disk
- `image_bgr`: lazily cached BGR image shared by tasks of the same frame
- `track_result`: current tracking result array, usually MOT-style `[frame, id, x, y, w, h, score, cls, vis]`
- `tag`: optional frame tag used for file naming

`FrameContext` also provides:

```python
frame.get_image_bgr() -> Optional[np.ndarray]
```

This loads the original frame at most once and reuses it across tasks.

### 2.2 HookEvent

Defined in [core.py](../core.py).

```python
HookEvent(
    name: str,
    switch: str,
    seq: int,
    timestamp: float,
    payload: dict[str, Any],
)
```

Main fields:
- `name`: logical event name used by tasks
- `switch`: hook switch name in config
- `seq`: event sequence number
- `timestamp`: capture timestamp
- `payload`: snapshot data captured by `TensorHook`

### 2.3 BaseVisualTask lifecycle

Defined in [core.py](../core.py).

Each task follows this lifecycle:

1. `__init__(task_name, cfg, mode, root_dir)`
2. `bind_image_writer(writer)`
3. `init()`
4. `update(frame, hook_events)` for each frame
5. `close()` at the end of the whole run

Optional capability methods:
- `required_switches() -> set[str]`
  Declares `TensorHook` switches required by this task.
- `required_time_switches() -> set[str]`
  Declares `GetTime` metric names required by this task.
- `requires_image() -> bool`
  Tells `Visualizer` whether this task participates in the shared image-loading path.

Important note:
- If any enabled task returns `True`, `Visualizer` will load the frame through `FrameContext.get_image_bgr()`.
- Frame image export should use the shared writer via `BaseVisualTask.submit_image(...)`.
- Video-level state should be keyed by `frame.video_name` instead of depending on frame order.

## 3. How tasks are configured

Tasks are configured under `Visualizer.tasks` in YAML. The standard dataset configuration is [dancetrack_train.yaml](../../../configs/dancetrack_train.yaml), while specialized offline scripts enable only the tasks they require at runtime.

```yaml
Visualizer:
  enabled: true
  hook:
    enabled: true
    switches:
      hqg_topk_source: true
      newborn_selector: true
      det_recover_monitor: true
  time:
    enabled: false
    switches:
      Model_forward: true
  tasks:
    bbox_render:
      enabled: false
    hqg_topk_roi_map:
      enabled: false
    hqg_histogram:
      enabled: false
    det_recover_monitor:
      enabled: true
    runtime_profile:
      enabled: false
```

Auto-enable behavior:
- If a task is enabled, `Visualizer` will automatically enable the hook switches returned by `required_switches()`.
- If a task is enabled, `Visualizer` will automatically enable the timing names returned by `required_time_switches()`.

## 4. Task reference

### 4.1 `bbox_render`

File:
- [bbox_render.py](bbox_render.py)

Purpose:
- Draw final tracking boxes and track IDs on the image.
- Useful for basic qualitative inspection of final tracking output.

Dependencies:
- No hook event required.
- Uses `FrameContext.track_result`.

Expected `track_result` layout:
- column `1`: track id
- columns `2:6`: `bb_left, bb_top, bb_width, bb_height`

Main config keys:
- `enabled`
- `save_image`
- `show_image`
- `window_delay`
- `draw_frame_text`

Outputs:
- Per-video images under:
  - `<visualize_root>/<video_name>/bbox_render/*.jpg`

When to use:
- Quick sanity check for final result quality.
- Human review of ID continuity and box quality.

Notes:
- `requires_image()` should return `True` whenever the task needs the image for save/show.
- The task receives the shared lazy-loaded frame image via `frame.get_image_bgr()`.

### 4.2 `hqg_topk_roi_map`

File:
- [hqg_topk_roi_map.py](hqg_topk_roi_map.py)

Purpose:
- Visualize the spatial coverage of decoder input queries as a heat overlay.
- Optionally overlay tracker boxes and/or query boxes.
- Helps inspect where HQG is focusing on the image.

Required hook switch:
- `hqg_topk_source`

Expected hook payload:
- `main_boxes`
- `main_logits`
- `main_mask`

Input interpretation:
- `main_boxes`: normalized `cxcywh` boxes
- `main_logits`: decoder logits used to derive score
- `main_mask`: invalid query mask

Main config keys:
- `enabled`
- `save_image`
- `show_image`
- `window_delay`
- `alpha`
- `min_score`
- `draw_frame_text`
- `draw_track_bbox`
- `draw_query_bbox`
- `track_bbox_alpha`

Outputs:
- Per-video images under:
  - `<visualize_root>/<video_name>/hqg_topk_roi_map/*.jpg`

When to use:
- Inspect whether HQG query coverage is concentrated in meaningful regions.
- Compare detector/query source focus against tracker output.

Notes:
- Only valid queries are rendered.
- Heat intensity is score-weighted by the max sigmoid score of each query.

### 4.3 `hqg_histogram`

File:
- [hqg_histogram.py](hqg_histogram.py)

Purpose:
- Aggregate score histograms for:
  - HQG top-k query scores
  - high-confidence newborn candidate scores
- Useful for comparing HQG source distribution with newborn selection distribution.

Required hook switches:
- `hqg_topk_source`
- `newborn_selector`

Expected hook payloads:

From `hqg_topk_source`:
- `main_logits`
- `main_mask`

From `newborn_selector`:
- `pred_logits`
- `pred_mask`
- `high_conf_index`

Main config keys:
- `enabled`
- `bins`
- `range`
- `save_png`
- `save_csv`
- `save_npz`

Outputs:
- Per-video:
  - `<visualize_root>/<video_name>/hqg_hist_topk_summary.png`
  - `<visualize_root>/<video_name>/hqg_hist_newborn_summary.png`
  - optional `hqg_hist_summary.csv`
  - optional `hqg_hist_summary.npz`
- Global:
  - `<visualize_root>/hqg_hist_topk_summary.png`
  - `<visualize_root>/hqg_hist_newborn_summary.png`
  - optional `hqg_hist_summary.csv`
  - optional `hqg_hist_summary.npz`

When to use:
- Tune newborn thresholds.
- Compare score distributions before and after HQG-related changes.

Notes:
- This task is summary-only and does not render per-frame images.
- Histograms are accumulated over frames and exported when a video ends or when the whole run ends.

### 4.4 `det_recover_monitor`

File:
- [det_recover_monitor.py](det_recover_monitor.py)

Purpose:
- Monitor direct detection-based recovery of `LOST` tracks.
- Visualize accepted recoveries and optionally rejected candidates.
- Export per-frame records and aggregate distributions for review.

Required hook switch:
- `det_recover_monitor`

Expected hook payload:
- `recover_vis`, which is a `list[dict]`, typically one item per batch entry

Expected fields inside each `recover_vis` item:
- `lost_track_ids`
- `best_det_indices`
- `best_det_boxes`
- `best_app_cos`
- `best_motion_cost`
- `best_total_cost`
- `best_age`
- `best_accepted`
- `best_reject_reason`
- `accepted_track_ids`
- `accepted_det_boxes`
- `accepted_app_cos`
- `accepted_motion_cost`
- `accepted_total_cost`
- `accepted_age`

Reject reason mapping:
- `0 -> accepted`
- `1 -> threshold`
- `2 -> assignment`

Main config keys:
- `enabled`
- `save_image`
- `show_image`
- `window_delay`
- `draw_frame_text`
- `save_csv`
- `save_png`
- `save_all_frames`
- `draw_rejected`

Outputs:
- Per-video:
  - `<visualize_root>/<video_name>/det_recover_monitor/*.jpg`
  - `<visualize_root>/<video_name>/det_recover_summary.csv`
  - `<visualize_root>/<video_name>/det_recover_summary.png`
- Global:
  - `<visualize_root>/det_recover_summary.csv`
  - `<visualize_root>/det_recover_summary.png`

CSV fields:
- `video_name`
- `frame_id`
- `track_id`
- `best_det_idx`
- `accepted`
- `reject_reason`
- `age`
- `app_cos`
- `motion_cost`
- `total_cost`

When to use:
- Audit whether detection-based re-capture is actually being triggered.
- Check whether appearance similarity is separable enough.
- Check whether current threshold mostly rejects by `threshold` or by `assignment`.
- Support threshold tuning on the full validation set.

Notes:
- Accepted recoveries are drawn in red.
- Rejected candidates are optional and only drawn when `draw_rejected=True`.
- If `save_all_frames=False`, only frames with accepted recoveries are saved by default.

### 4.5 `runtime_profile`

File:
- [runtime_profile.py](runtime_profile.py)

Purpose:
- Export runtime summary from `GetTime`.
- Useful for profiling key timed sections during inference or training.

Required time switches:
- The names listed in config `names`

Main config keys:
- `enabled`
- `names`
- `save_txt`
- `save_csv`

Outputs:
- `<visualize_root>/runtime_profile/runtime_profile_summary.txt`
- `<visualize_root>/runtime_profile/runtime_profile_summary.csv`

When to use:
- Compare latency before and after code changes.
- Track the cost of selected timed sections such as `Model_forward`.

Notes:
- `update()` is a no-op for this task.
- Data is exported only once at `close()`.

## 5. Current task selection summary

Current built-in task coverage:

| Task | Per-frame image | Per-video summary | Global summary | Hook dependent | Time dependent |
| --- | --- | --- | --- | --- | --- |
| `bbox_render` | Yes | No | No | No | No |
| `hqg_topk_roi_map` | Yes | No | No | Yes | No |
| `hqg_histogram` | No | Yes | Yes | Yes | No |
| `det_recover_monitor` | Optional | Yes | Yes | Yes | No |
| `runtime_profile` | No | No | Yes | No | Yes |

## 6. How to add a new task

Recommended minimal steps:

1. Add a new file under this directory, for example `my_task.py`.
2. Inherit `BaseVisualTask`.
3. Implement at least:
   - `__init__`
   - `update`
4. Implement optional methods if needed:
   - `required_switches`
   - `required_time_switches`
   - `requires_image`
   - `_finalize_video`
   - `close`
5. Register the class in:
   - [__init__.py](__init__.py)
   - [Visualizer._build_tasks in core.py](../core.py)
6. Add YAML config under `Visualizer.tasks`.
7. If the task needs hook data, ensure the producing module emits a matching `TensorHook` event and payload structure.

Recommended design rules:
- Keep one task focused on one review goal.
- Do not mix heavy rendering and heavy statistics into one task unless they always need to be enabled together.
- Prefer exporting review-friendly CSV and PNG when the task is intended for threshold tuning or ablation analysis.

## 7. Review checklist

When reviewing a task, the most important questions are:

1. Is the task goal single-purpose and clear?
2. Is the hook payload contract explicit and stable?
3. Are outputs review-friendly and easy to compare across runs?
4. Does the task avoid unnecessary image loading or large tensor copies?
5. Can the task be enabled independently without affecting other tasks?

This README should be updated whenever:
- a task is added or removed
- a hook payload schema changes
- a task output path or summary format changes
