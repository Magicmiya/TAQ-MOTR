# TAQ-MOTR Quick Start

本文档给出 TAQ-MOTR 在常用 MOT 数据集上的训练、验证、测试提交和结果定位流程。开始前请先完成 [INSTALL.md](INSTALL.md) 中的环境安装、CUDA 算子编译和数据集预处理。

## 1. 准备检查

请在项目根目录执行下面的检查。后续命令默认也都从项目根目录运行。

本文命令以项目根目录的标准入口 `main.py` 为例。执行前请确认当前 checkout 包含该入口文件；如果你的工作区缺少 `main.py`，需要先恢复完整代码入口，再运行下面的训练和评估命令。

```bash
conda activate TAQ_MOTR

# CUDA 和 PyTorch 基础检查
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# Deformable Attention CUDA ops 检查
python -c "import MultiScaleDeformableAttention as m; print(m.__file__)"
```

数据集目录需要先按 [INSTALL.md](INSTALL.md#dataset-prepare) 生成项目需要的辅助文件。配置文件中的 `Dataset.dataset_root` 应指向包含 `DanceTrack/`、`MOT17/`、`MOT20/` 或 `SportMOT/` 的数据集根目录。

## 2. 准备预训练权重

TAQ-MOTR 默认使用 COCO 上训练的 RT-DETRv2-L 权重初始化 Hybrid Encoder 相关特征抽取能力。推荐把权重放到项目根目录的 `weight/` 或输出目录外的稳定位置，然后在 YAML 或命令行里指定。

```text
weight/
`- rtdetrv2_r50vd_6x_coco_ema.pth
```

下载地址：

- RT-DETRv2-L: <https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_6x_coco_ema.pth>

训练时可以通过 `PRETRAINED_MODEL` 或命令行 `-P/--pretrained_model` 指定该文件。

## 3. 选择配置文件

常用配置位于 `configs/`：

| 场景 | 推荐配置 |
| --- | --- |
| DanceTrack 单卡训练 | `configs/dancetrack_train_single_gpu.yaml` |
| DanceTrack 多卡训练 | `configs/dancetrack_train_multi_gpu.yaml` |
| DanceTrack 验证/测试 | `configs/dancetrack_eval_single_gpu.yaml` |
| SportsMOT 多卡训练 | `configs/sportsmot_train_multi_gpu.yaml` 或 `configs/sportsmot_train_multi_gpu_new.yaml` |
| SportsMOT 验证/测试 | `configs/sportsmot_eval_single_gpu.yaml` |
| MOT17 多卡训练 | `configs/mot17_train_multi_gpu.yaml` |
| MOT17 验证/测试 | `configs/mot17_eval_single_gpu.yaml` |
| MOT20 多卡训练 | `configs/mot20_train_multi_gpu.yaml` |
| MOT20 验证/测试 | `configs/mot20_eval_single_gpu.yaml` |

最常改的字段如下，详细含义见 [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md)。

```yaml
MODE: train
OUTPUTS_DIR: outputs/
PRETRAINED_MODEL: weight/rtdetrv2_r50vd_6x_coco_ema.pth

Training:
  Available_gpus: 0,
  use_distributed: false

Dataset:
  dataset_name: DanceTrack
  dataset_root: /path/to/datasets/root
```

命令行参数优先级高于 YAML，因此临时改路径或 GPU 时不必复制配置文件：

```bash
python main.py -E DanceTrack_train \
  -C configs/dancetrack_train_single_gpu.yaml \
  -DR /path/to/datasets/root \
  --gpus 0 \
  -P weight/rtdetrv2_r50vd_6x_coco_ema.pth
```

## 4. 训练

### 单卡训练

```bash
python main.py \
  -E DanceTrack_single_gpu \
  -C configs/dancetrack_train_single_gpu.yaml \
  -DR /path/to/datasets/root \
  --gpus 0 \
  -P weight/rtdetrv2_r50vd_6x_coco_ema.pth
```

### 多卡训练

多卡训练需要同时使用 `torch.distributed.run` 和 `--distributed`，并保持 `--nproc_per_node` 与 `--gpus` 的 GPU 数量一致。

```bash
OMP_NUM_THREADS=1 python -m torch.distributed.run --nproc_per_node=4 main.py \
  -E DanceTrack_multi_gpu \
  -C configs/dancetrack_train_multi_gpu.yaml \
  -DR /path/to/datasets/root \
  --gpus 0,1,2,3 \
  --distributed \
  -P weight/rtdetrv2_r50vd_6x_coco_ema.pth
```

训练输出默认写入：

```text
outputs/<EXP_NAME>[_timestamp]/
|- train/
|  |- config.yaml
|  |- log.txt
|  `- tb_*
|- checkpoint_last.pth
`- checkpoint_best_*.pth
```

`train/config.yaml` 是本次运行最终合并后的完整配置，验证和测试会优先从 checkpoint 同级目录下的 `train/config.yaml` 读取模型结构参数。

## 5. 恢复训练

从已有 checkpoint 继续训练：

```bash
python main.py \
  -E DanceTrack_resume \
  -C configs/dancetrack_train_single_gpu.yaml \
  -R outputs/DanceTrack_single_gpu/checkpoint_last.pth \
  -DR /path/to/datasets/root \
  --gpus 0
```

如果 `RESUME` 指向当前 `OUTPUTS_DIR` 下的 checkpoint，会继续写入原运行目录；如果指向其他位置，程序会创建新的 resume 输出目录并复制原训练日志目录。

## 6. 验证

验证需要设置 `MODE=val`，并通过 `-EFP/--eval_file_path` 指向单个 checkpoint 或 checkpoint 目录。推荐使用 runtime-only eval 配置：

```bash
python main.py \
  -E DanceTrack_val \
  -M val \
  -C configs/dancetrack_eval_single_gpu.yaml \
  -EFP outputs/DanceTrack_single_gpu/checkpoint_last.pth \
  -DR /path/to/datasets/root \
  --gpus 0
```

只验证少量视频用于快速检查：

```bash
python main.py \
  -E DanceTrack_smoke \
  -M val \
  -C configs/dancetrack_eval_single_gpu.yaml \
  -EFP outputs/DanceTrack_single_gpu/checkpoint_last.pth \
  -DR /path/to/datasets/root \
  --gpus 0 \
  -V dancetrack0041,dancetrack0043
```

验证输出默认写入 checkpoint 根目录：

```text
outputs/<TRAIN_EXP>/
`- val/
   |- inference_val/
   |  `- <EXP_NAME>_<checkpoint_name>/
   |- eval_log.txt
   `- trackeval_summary.txt
```

## 7. 测试和提交文件

测试集推理设置 `MODE=test`。程序会生成 benchmark 可提交的 zip 文件。

```bash
python main.py \
  -E DanceTrack_test \
  -M test \
  -C configs/dancetrack_eval_single_gpu.yaml \
  -EFP outputs/DanceTrack_single_gpu/checkpoint_last.pth \
  -DR /path/to/datasets/root \
  --gpus 0
```

测试输出默认位于：

```text
outputs/<TRAIN_EXP>/
`- test/
   `- inference_test/
      |- <EXP_NAME>_<checkpoint_name>/
      |  |- result/
      |  `- config.yaml
      `- submission_<EXP_NAME>_<checkpoint_name>.zip
```

MOT17/MOT20 测试会按 MOTChallenge 提交格式额外整理 `submission_txt/`；DanceTrack 和 SportsMOT 直接打包跟踪结果 txt。

## 8. 批量配置和消融

`-C/--config_path` 支持单个 YAML、逗号分隔的多个 YAML，或包含 YAML 的目录。多个配置会按文件名为实验名追加配置后缀。

```bash
python main.py \
  -E DanceTrack_ablation \
  -C configs/dancetrack_ablation \
  -M val \
  -EFP outputs/DanceTrack_single_gpu/checkpoint_last.pth \
  -DR /path/to/datasets/root \
  --gpus 0
```

`--sweep` 支持对 YAML 内的嵌套字段做笛卡尔积展开：

```bash
python main.py \
  -E DanceTrack_lcm_scan \
  -C configs/dancetrack_eval_single_gpu.yaml \
  -M val \
  -EFP outputs/DanceTrack_single_gpu/checkpoint_last.pth \
  --sweep Life_cycle_management.track_thresh=0.45,0.5,0.55 Inference.result_score_thresh=0.4,0.5
```

不允许在一次 sweep 中覆盖运行目录、数据根目录、GPU 和分布式开关等运行级字段。需要改这些字段时请使用普通命令行参数或复制 YAML。

## 9. 常见问题

### 找不到数据集

确认 `-DR/--dataset_root` 或 `Dataset.dataset_root` 指向数据集根目录，而不是某个 split 目录。例如应指向 `/data/MOT`，其中包含 `/data/MOT/DanceTrack/train`、`/data/MOT/DanceTrack/val`、`/data/MOT/DanceTrack/test`。

### 验证时报缺少模型结构键

验证配置通常只保存运行时参数。程序会从 `<checkpoint_root>/train/config.yaml` 补齐模型结构。如果该文件不存在，需要恢复训练输出目录里的 `train/config.yaml`，或使用包含完整模型结构的 YAML 进行验证。

### 多卡输出分散或重复

多卡训练请使用 `torch.distributed.run`、`--distributed` 和正确的 `--gpus`。程序会在分布式初始化后同步同一个输出目录，避免不同 rank 写到不同时间戳目录。

### 结果阈值与生命周期阈值不同

`Inference.result_score_thresh` 控制最终 txt 导出的结果过滤；`Life_cycle_management.track_thresh`、`new_born_threshold`、`high_conf_threshold` 控制在线跟踪生命周期。调验证指标时建议优先在 eval YAML 或 `--sweep` 中调整。
