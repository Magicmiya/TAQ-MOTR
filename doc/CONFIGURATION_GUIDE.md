# TAQ-MOTR Configuration Guide

本文档说明 TAQ-MOTR 的命令行参数、YAML 配置结构和常见修改方式。快速运行流程见 [QUICK_START.md](QUICK_START.md)。

## 1. 配置加载规则

运行配置由两部分合并得到：

1. `-C/--config_path` 指定的 YAML。
2. 命令行参数。

优先级为：

```text
command line > YAML
```

最终配置会在训练时写入：

```text
<OUTPUTS_DIR>/train/config.yaml
```

验证和测试时，如果 `EVAL_FILE_PATH` 指向 checkpoint 或 checkpoint 目录，程序会优先读取：

```text
<checkpoint_root>/train/config.yaml
```

这样可以保证评估时使用与训练 checkpoint 匹配的模型结构，同时仍允许 eval YAML 覆盖部分推理参数。

## 2. 命令行参数

命令行参数在 `configs/utils.py` 中定义，常用参数如下：

下面示例以项目根目录的标准入口 `main.py` 为例。执行命令前请确认当前 checkout 包含该入口文件。

| 参数 | 必填 | 作用 |
| --- | --- | --- |
| `-E`, `--exp_name` | 是 | 实验名。输出目录和 sweep 后缀会基于它生成。 |
| `-C`, `--config_path` | 是 | YAML 文件、逗号分隔 YAML 列表，或包含 YAML 的目录。相对路径会优先从 `configs/` 下解析。 |
| `-M`, `--mode` | 否 | 运行模式，常用 `train`、`val`、`test`。命令行会覆盖 YAML 的 `MODE`。 |
| `-O`, `--outputs_dir` | 否 | 输出根目录。相对路径按项目根目录解析。 |
| `-NEW`, `--new_out_dir` | 否 | 是否在训练输出目录后追加时间戳。YAML 中通常用 `NEW_OUT_DIR` 控制。 |
| `-P`, `--pretrained_model` | 否 | 预训练权重路径，训练初始化时使用。 |
| `-L`, `--checkpoint_level` | 否 | 梯度 checkpoint 级别。`-1` 表示根据 batch 中帧数自动选择。 |
| `-R`, `--resume` | 否 | 恢复训练 checkpoint 路径。 |
| `-V`, `--videos` | 否 | 验证/测试视频子集，逗号分隔，例如 `dancetrack0041,dancetrack0043`。 |
| `--gpus` | 否 | GPU id 列表，例如 `0` 或 `0,1,2,3`。覆盖 `Training.Available_gpus`。 |
| `--distributed` | 否 | 启用分布式训练或分布式评估。需要配合 `torch.distributed.run`。 |
| `-EFP`, `--eval_file_path` | 否 | 待评估 checkpoint 文件或包含多个 `.pth` 的目录。覆盖 `EVAL_FILE_PATH`。 |
| `-DR`, `--dataset_root` | 否 | 数据集根目录。覆盖 `Dataset.dataset_root`。 |
| `--sweep` | 否 | 运行时消融扫描，格式为 `Section.key=v1,v2`，支持多个字段笛卡尔积组合。 |

示例：

```bash
python main.py \
  -E DanceTrack_val \
  -M val \
  -C configs/dancetrack_eval_single_gpu.yaml \
  -EFP outputs/DanceTrack_train/checkpoint_last.pth \
  -DR /path/to/datasets/root \
  --gpus 0 \
  -V dancetrack0041,dancetrack0043
```

## 3. `config_path` 解析方式

`-C/--config_path` 支持三种写法。

### 单个 YAML

```bash
python main.py -E exp -C configs/dancetrack_train_single_gpu.yaml
```

也可以只写相对 `configs/` 的路径：

```bash
python main.py -E exp -C dancetrack_train_single_gpu.yaml
```

### 多个 YAML

```bash
python main.py -E exp -C configs/a.yaml,configs/b.yaml
```

多个配置会逐个运行，实验名会自动追加配置文件名后缀。

### YAML 目录

```bash
python main.py -E exp -C configs/dancetrack_ablation
```

目录内所有 `.yaml` 和 `.yml` 文件会按文件名排序后逐个运行。

## 4. 顶层运行字段

每个训练 YAML 通常包含以下顶层字段：

```yaml
EXP_NAME: Dancetrack_train_single_gpu
MODULE_NAME: FMI_MOTR
CONFIG_PATH:
MODE: train
OUTPUTS_DIR: outputs/
NEW_OUT_DIR: false
PRETRAINED_MODEL: weight/rtdetrv2_r50vd_6x_coco_ema.pth
CHECKPOINT_LEVEL: 1
RESUME:
EVAL_FILE_PATH:

hidden_dim: 256
num_classes: 1
visualize: false
batch_norm: normal
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `EXP_NAME` | 实验名。命令行 `-E` 会覆盖它。 |
| `MODULE_NAME` | 模型构建入口名称，当前为 `FMI_MOTR`。 |
| `CONFIG_PATH` | 运行后会记录实际使用的 YAML 路径。通常无需手动设置。 |
| `MODE` | `train`、`val` 或 `test`。 |
| `OUTPUTS_DIR` | 输出根目录。训练时通常生成 `outputs/<EXP_NAME>/` 或带时间戳目录。 |
| `NEW_OUT_DIR` | 训练时是否自动追加时间戳。 |
| `PRETRAINED_MODEL` | 训练初始化权重路径。验证/测试通常为空。 |
| `CHECKPOINT_LEVEL` | 显存节省级别，数值越大 checkpoint 范围越广，速度越慢。 |
| `RESUME` | 恢复训练 checkpoint。 |
| `EVAL_FILE_PATH` | 验证/测试 checkpoint 文件或目录。 |
| `hidden_dim` | 模型隐藏维度，需要与各模块配置保持一致。 |
| `num_classes` | 数据集类别数。DanceTrack/SportsMOT 通常为 1。 |
| `visualize` | 是否启用可视化 hook 和任务。 |
| `batch_norm` | BN 策略，例如 `normal`、`freeze`、`no_local`。 |

## 5. Training 配置

`Training` 控制 GPU、优化器、训练阶段、日志和恢复策略。

```yaml
Training:
  Available_gpus: 0,
  use_distributed: false
  eval_after_epoch: 1
  seed: 42
  device: cuda

  epochs: 30
  lr_rate:
    backbone: 2.0e-5
    sampling_offsets: 1.0e-5
    query_updater: 2.0e-4
    default: 2.0e-4

  weight_decay: 0.0005
  lr_scheduler: MultiStep
  lr_drop_milestones: [21]
  lr_drop_rate: 0.1
  max_train_iters:
  multi_checkpoint: false
  resume_scheduler: true
  clip_max_norm: 0.1
  accumulation_steps: 1
```

重点字段：

| 字段 | 说明 |
| --- | --- |
| `Available_gpus` | 可见 GPU id 字符串。命令行 `--gpus` 会覆盖。 |
| `use_distributed` | 是否使用分布式运行。命令行 `--distributed` 会设为 true。 |
| `eval_after_epoch` | 训练过程中每隔多少 epoch 做一次验证。 |
| `epochs` | 总训练 epoch。 |
| `lr_rate` | 分模块学习率。`default` 必须保留。 |
| `stage_policy` | 多阶段 clip 长度、batch size、采样间隔和损失权重策略。 |
| `max_train_iters` | 快速 smoke 训练时限制每个 epoch 的迭代数，正式训练保持为空。 |
| `resume_scheduler` | 恢复训练时是否恢复优化器和调度相关状态。 |
| `accumulation_steps` | 梯度累积步数。显存不足时可增大。 |

`stage_policy` 示例：

```yaml
Training:
  stage_policy:
    sample_steps: [0, 8, 14, 18, 21, 26]
    stages:
      - sample_length: 2
        batch_size: 5
        sample_mode: random_interval
        sample_interval: 10
        use_aux_loss: true
        use_det_dn_aux: true
        use_dn: true
        high_conf_threshold: 0.5
        loss_tqi: 0.0
      - sample_length: 3
        batch_size: 4
        high_conf_threshold: 0.55
```

训练时会根据当前 epoch 选择对应 stage，并同步更新 dataset、model 和部分 loss/Life-cycle 配置。

## 6. Dataset 配置

```yaml
Dataset:
  dataset_name: DanceTrack
  dataset_root: /path/to/datasets/root
  sample_length: 2
  batch_size: 5
  sample_mode: random_interval
  sample_interval: 10
  sampler_shuffle: false
  num_workers: 4
  persistent_workers: true
  prefetch_factor: 2
  coco_size: false
  overflow_bbox: false
  reverse_clip: false
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `dataset_name` | 数据集名称。当前构建器支持 `DanceTrack`、`SportsMOT`、`MOT17`、`MOT20`、`CrowdHuman`、`CityPersons`。 |
| `dataset_root` | 数据集根目录。命令行 `-DR` 会覆盖。 |
| `train_datasets` | 可选。用于混合训练时声明多个训练数据集。 |
| `train_sub_sets` | 可选。训练时使用的数据 split，例如 DanceTrack 默认使用 `train`。 |
| `eval_videos` | 可选。验证/测试视频子集，命令行 `-V` 会覆盖。 |
| `sample_length` | 每个样本包含的帧数。训练时常由 `stage_policy` 覆盖。 |
| `batch_size` | dataloader batch size。训练时常由 `stage_policy` 覆盖。 |
| `sample_mode` | `random_interval` 或 `fixed_interval`。 |
| `sample_interval` | 最大采样间隔。 |
| `num_workers` | dataloader worker 数。 |
| `coco_size`、`overflow_bbox`、`reverse_clip` | 数据增强相关开关。 |

数据集根目录示例：

```text
/path/to/datasets/root/
|- DanceTrack/
|  |- train/
|  |- val/
|  `- test/
|- MOT17/
|  |- annotations/
|  |- train/
|  `- test/
`- SportMOT/
   `- dataset/
      |- train/
      `- test/
```

## 7. Backbone 和 Encoder 配置

Backbone 示例：

```yaml
Backbone:
  depth: 50
  variant: d
  freeze_at: 1
  return_idx: [1, 2, 3]
  num_stages: 4
  freeze_norm: true
  pretrained: false
```

Encoder 示例：

```yaml
Encoder:
  feature_levels: 3
  in_channels: [512, 1024, 2048]
  feat_strides: [8, 16, 32]
  use_group_norm: false
  use_padding_mask: true
  hidden_dim: 256
  use_encoder_idx: [2]
  num_encoder_layers: 1
  nhead: 8
  dim_feedforward: 1024
  dropout: 0.0
  enc_act: gelu
  pe_temperature: 20
  batch_norm: normal
```

这些字段定义模型结构，通常必须与 checkpoint 训练时一致。验证/测试时如果 checkpoint 根目录存在 `train/config.yaml`，程序会从其中恢复结构字段。

## 8. Decoder 配置

`Decoder` 是 TAQ-MOTR/HQG 相关配置最集中的部分。

```yaml
Decoder:
  num_classes: 1
  norm_style: freeze_BN
  activation: gelu

  det_query_mode: hybrid
  query_select_method: default
  num_denoising: 200
  hqg_init_feat_levels: [-1]
  hqg_num_blocks: 3
  hqg_num_learnable_memory: 32
  hqg_active_learnable_memory: null
  qpn_interact_max_state: 1
  hqg_det_dn_mask_track_memory: true
  use_det_dn: true

  feat_channels: [256, 256, 256]
  feat_strides: [8, 16, 32]
  feat_levels: 3
  hidden_dim: 256
  num_points: [4, 4, 4]

  use_sine_pos: true
  use_query_scale: true
  num_layers: 6
  nhead: 8
  num_queries: 200
  merge_det_track_layer: 1
  cross_attn_method: CUDA
  eval_idx: -1
```

常见调整：

| 字段 | 说明 |
| --- | --- |
| `num_queries` | 检测 query 数量。验证时允许通过 eval YAML 覆盖。 |
| `hqg_init_feat_levels` | HQG seed feature level。`-1` 表示最后一层特征。 |
| `hqg_num_blocks` | HQG 交互 block 数。 |
| `hqg_num_learnable_memory` | HQG learnable memory 参数容量。 |
| `hqg_active_learnable_memory` | 前向实际启用的 memory token 数。`null` 表示全部启用。 |
| `qpn_interact_max_state` | HQG track memory 允许交互的最大 track 状态。`-1` 表示关闭 track memory。 |
| `merge_det_track_layer` | 从第几层 decoder 开始合并 detection query 和 track query。 |
| `cross_attn_method` | `CUDA` 使用编译后的 deformable attention CUDA 实现；其他值会走 PyTorch 路径。 |
| `eval_idx` | 推理使用的 decoder 层，`-1` 表示最后一层。 |

## 9. Criterion 配置

`Criterion` 控制匹配器、主损失和辅助损失权重。

```yaml
Criterion:
  num_classes: 1
  hidden_dim: 256
  frame_length: [2, 3, 4, 5, 7]
  num_decoder_layer: 6
  merge_det_track_layer: 1
  alpha: 0.25
  gamma: 2.0
  losses_weight:
    loss_bboxes_L1: 5
    loss_bboxes_giou: 2
    loss_labels_vfl: 2
    loss_topk_disp: 0.2
    loss_tqi: 0.0
  tqi_tau: 0.07
  topk_disp_enable: true
  topk_disp_min_iou: 0.7
  dn_num: 200
  aux_loss: true
  aux_weights: [1.0, 1.0, 1.0, 1.0, 1.0]
  det_dn_aux_weights: [0.25, 0.25, 0.25, 0.5, 0.75, 1.0]
```

训练阶段中 `stage_policy` 可以动态覆盖部分 loss 相关字段，例如 `loss_tqi`、`use_aux_loss`、`use_det_dn_aux` 和 `use_dn`。

## 10. Inference 配置

`Inference` 控制结果导出，不等同于内部生命周期管理。

```yaml
Inference:
  result_score_thresh: 0.5
  result_min_area: 100
  result_only_active: true
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `result_score_thresh` | 导出 txt 前的分数阈值。 |
| `result_min_area` | 导出 txt 前的最小框面积过滤。 |
| `result_only_active` | 是否只导出处于 active 状态的 track。 |

调 benchmark 指标时，这些字段适合用 eval YAML 或 `--sweep` 扫描。

## 11. Life_cycle_management 配置

`Life_cycle_management` 控制在线跟踪状态转换、新生轨迹、消亡和恢复策略。

```yaml
Life_cycle_management:
  hidden_dim: 256
  num_classes: 1
  high_conf_threshold: 0.5
  new_born_threshold: 0.9
  track_thresh: 0.5
  miss_tolerance: 30
  Sudden_death_threshold: 0.5
  long_memory_lambda: 0.01
  tp_drop_ratio: 0.0
  fp_insert_ratio: 0.0
  no_tracking_augment: true
  recover_iou_threshold: 0.5
  det_recover_enable: true
  det_recover_max_time: 8
  det_recover_min_history: 2
  det_recover_app_weight: 0.55
  det_recover_motion_weight: 0.45
  det_recover_cost_threshold: 0.4
  det_recover_center_sigma: 0.35
  det_recover_shape_sigma: 0.12
```

常见调整：

| 字段 | 说明 |
| --- | --- |
| `high_conf_threshold` | 高置信轨迹阈值。训练 stage 可覆盖。 |
| `new_born_threshold` | 新生轨迹阈值。 |
| `track_thresh` | 在线跟踪保留阈值。 |
| `miss_tolerance` | track 可容忍的连续丢失帧数。 |
| `Sudden_death_threshold` | 低置信快速消亡阈值。 |
| `det_recover_enable` | 是否启用检测恢复策略。 |
| `det_recover_*` | 检测恢复的时间、历史、外观/运动权重和代价阈值。 |

注意：`track_thresh` 会影响内部轨迹生命周期；`Inference.result_score_thresh` 只影响最终结果导出。

## 12. Query_updater 配置

```yaml
Query_updater:
  hidden_dim: 256
  ffn_dim: 2048
  dropout: 0.0
  tp_drop_ratio: 0.0
  fp_insert_ratio: 0.0
  no_tracking_augment: true
  long_memory_lambda: 0.01
  use_dab: true
  use_sine_pos: true
```

这些字段属于模型结构和训练行为配置。除非明确做消融，验证/测试时应保持与 checkpoint 训练配置一致。

## 13. Visualizer 配置

`visualize: true` 时会启用 `Visualizer`。`Visualizer.tasks` 下的各任务可以独立开关。

```yaml
visualize: true

Visualizer:
  enabled: true
  workers: 4
  writer_workers: 2
  show_cross_attn: true
  show_self_attn: true
  show_first_layer: true
  hook:
    enabled: true
    queue_size: 2048
    block_on_full: true
    clone_tensor: true
    switches:
      hqg_topk_source: true
      newborn_selector: true
  tasks:
    bbox_render:
      enabled: true
      save_image: true
      show_image: false
    hqg_topk_roi_map:
      enabled: false
      save_image: false
    hqg_histogram:
      enabled: false
      save_png: false
      save_csv: false
```

可视化会增加显存、CPU 和磁盘开销。正式训练或全量验证默认建议关闭；需要分析注意力或恢复行为时再开启指定任务。

## 14. Eval 配置继承规则

`configs/dancetrack_eval_single_gpu.yaml` 和 `configs/sportsmot_eval_single_gpu.yaml` 属于 runtime-only eval 配置。它们通常只包含：

- 运行字段，例如 `MODE`、`EVAL_FILE_PATH`、`Training.Available_gpus`。
- 数据集路径和 eval 子集。
- 推理导出阈值。
- 少量允许验证时覆盖的 decoder 和 life-cycle 参数。

当 `MODE` 是 `val` 或 `test` 时，程序会：

1. 根据 `EVAL_FILE_PATH` 找到 checkpoint 根目录。
2. 尝试读取 `<checkpoint_root>/train/config.yaml`。
3. 从训练配置中恢复模型结构字段。
4. 保留 eval YAML 中允许覆盖的推理字段。

允许 eval YAML 覆盖的 `Decoder` 字段包括：

```text
num_queries
hqg_init_feat_level
hqg_init_feat_levels
qpn_interact_max_state
hqg_num_learnable_memory
hqg_active_learnable_memory
hqg_num_blocks
hqg_det_dn_mask_track_memory
```

如果 checkpoint 根目录缺少 `train/config.yaml`，eval YAML 必须自己包含完整模型结构字段，否则会报缺少模型配置键。

## 15. Sweep 消融

`--sweep` 可以在不复制 YAML 的情况下扫描嵌套字段。

```bash
python main.py \
  -E DanceTrack_scan \
  -C configs/dancetrack_eval_single_gpu.yaml \
  -M val \
  -EFP outputs/DanceTrack_train/checkpoint_last.pth \
  --sweep Decoder.hqg_num_learnable_memory=16,32 Life_cycle_management.track_thresh=0.45,0.5
```

上面的命令会展开为 4 个组合，实验名会自动追加字段缩写和值。常见缩写包括：

| 字段 | 后缀缩写 |
| --- | --- |
| `Decoder.hqg_num_learnable_memory` | `LM` |
| `Decoder.hqg_active_learnable_memory` | `ALM` |
| `Decoder.hqg_num_blocks` | `HB` |
| `Decoder.qpn_interact_max_state` | `QIS` |
| `Decoder.num_queries` | `NQ` |
| `Life_cycle_management.new_born_threshold` | `NB` |
| `Life_cycle_management.track_thresh` | `TT` |
| `Life_cycle_management.high_conf_threshold` | `HC` |
| `Life_cycle_management.Sudden_death_threshold` | `SD` |
| `Inference.result_score_thresh` | `RS` |
| `Inference.result_min_area` | `RMA` |

以下字段不允许通过 `--sweep` 覆盖：

```text
EXP_NAME
CONFIG_PATH
MODE
OUTPUTS_DIR
NEW_OUT_DIR
Training.Available_gpus
Training.use_distributed
Dataset.dataset_root
```

这些字段属于一次运行的外部环境，应通过普通命令行参数或单独 YAML 控制。

## 16. 推荐修改方式

### 临时换数据路径或 GPU

优先用命令行：

```bash
python main.py -E exp -C configs/dancetrack_train_single_gpu.yaml -DR /data/MOT --gpus 1
```

### 长期保留一个实验设置

复制一个 YAML 并修改：

```bash
cp configs/dancetrack_train_single_gpu.yaml configs/my_dancetrack_train.yaml
```

然后固定写入数据集、训练阶段和模型消融参数。

### 调验证阈值

优先复制 eval YAML 或使用 `--sweep`，不要改训练输出目录下的 `train/config.yaml`。`train/config.yaml` 是 checkpoint 的复现实验记录。

### 多卡训练

YAML 中可设置：

```yaml
Training:
  Available_gpus: 0,1,2,3
  use_distributed: true
```

命令行中仍建议显式指定：

```bash
OMP_NUM_THREADS=1 python -m torch.distributed.run --nproc_per_node=4 main.py \
  -E exp \
  -C configs/dancetrack_train_multi_gpu.yaml \
  --gpus 0,1,2,3 \
  --distributed
```

## 17. 输出目录规则

训练：

```text
<OUTPUTS_DIR>/<EXP_NAME>[_timestamp]/
|- train/config.yaml
|- checkpoint_last.pth
`- checkpoint_best_*.pth
```

验证：

```text
<checkpoint_root>/val/inference_val/<EXP_NAME>_<checkpoint_name>/
```

测试：

```text
<checkpoint_root>/test/inference_test/<EXP_NAME>_<checkpoint_name>/
<checkpoint_root>/test/inference_test/submission_<EXP_NAME>_<checkpoint_name>.zip
```

其中 `<checkpoint_root>` 是 `EVAL_FILE_PATH` 指向的 checkpoint 文件所在目录，或 checkpoint 目录本身。
