# TAQ-MOTR Quick Start

This guide covers training, validation, test inference, and visualization. Complete the environment setup, CUDA op build, and dataset preprocessing in [INSTALL.md](INSTALL.md) first.

## 1. Sanity Checks

Run all commands from the repository root:

```bash
conda activate TAQ_MOTR
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import MultiScaleDeformableAttention as m; print(m.__file__)"
```

The dataset-side auxiliary files listed in [INSTALL.md](INSTALL.md#dataset-prepare) must exist before training or evaluation.

## 2. Pretrained Weights

TAQ-MOTR uses the COCO-pretrained RT-DETRv2 PResNet-50vd model:

- [rtdetrv2_r50vd_6x_coco_ema.pth](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_6x_coco_ema.pth)

The default repository-relative location is:

```text
weight/
`- rtdetrv2_r50vd_6x_coco_ema.pth
```

Relative filenames passed to `-P/--pretrained_model` are resolved from `weight/` automatically, so `-P rtdetrv2_r50vd_6x_coco_ema.pth` is sufficient.

## 3. Main Configs

Each recipe uses one complete YAML for training, validation, and test inference:

| Recipe | Config | Training data |
| --- | --- | --- |
| DanceTrack | `configs/dancetrack_train.yaml` | DanceTrack train |
| BFT | `configs/bft_train.yaml` | BFT train |
| SportsMOT | `configs/sportsmot_train.yaml` | SportsMOT train |
| SportsMOT Mix | `configs/sportsmot_mix.yaml` | SportsMOT train+val and CrowdHuman train+val |

See [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md) for configuration fields and runtime overrides.

## 4. Training

The following examples use four GPUs. Replace the dataset root placeholder before running a command.

### DanceTrack

```bash
OMP_NUM_THREADS=8 python -m torch.distributed.run --nproc_per_node=4 main.py \
  -E DanceTrack_train -C configs/dancetrack_train.yaml --gpus 0,1,2,3 --distributed \
  -DR <Path to your dataset root> -P rtdetrv2_r50vd_6x_coco_ema.pth
```

### BFT

Run `data/tools/gen_bft.py` as described in [Dataset Prepare](INSTALL.md#bft-preprocessing) before training:

```bash
OMP_NUM_THREADS=8 python -m torch.distributed.run --nproc_per_node=4 main.py \
  -E BFT_train -C configs/bft_train.yaml --gpus 0,1,2,3 --distributed \
  -DR <Path to your dataset root> -P rtdetrv2_r50vd_6x_coco_ema.pth
```

### SportsMOT

```bash
OMP_NUM_THREADS=8 python -m torch.distributed.run --nproc_per_node=4 main.py \
  -E SportsMOT_train -C configs/sportsmot_train.yaml --gpus 0,1,2,3 --distributed \
  -DR <Path to your dataset root> -P rtdetrv2_r50vd_6x_coco_ema.pth
```

### SportsMOT Mix

```bash
OMP_NUM_THREADS=8 python -m torch.distributed.run --nproc_per_node=4 main.py \
  -E SportsMOT_mix -C configs/sportsmot_mix.yaml --gpus 0,1,2,3 --distributed \
  -DR <Path to your dataset root> -P rtdetrv2_r50vd_6x_coco_ema.pth
```

Default training outputs:

```text
outputs/<EXP_NAME>[_timestamp]/
|- train/
|  |- config.yaml
|  |- log.txt
|  `- tb_*
|- checkpoint_last.pth
`- checkpoint_best_*.pth
```

`train/config.yaml` records the fully merged runtime configuration used by that checkpoint.

## 5. Reproduce Inference Results

To reproduce our reported results, download the released weights from the [Performance section of README](../README.md#-performance).

```bash
OMP_NUM_THREADS=1 python -m torch.distributed.run --nproc_per_node=4 main.py \
  -E <Experiment name> -M <val or test> -C <Path to the main config> --gpus 0,1,2,3 --distributed \
  -DR <Path to your dataset root> -EFP <Path to the checkpoint or checkpoint directory>
```

- `-M val` runs inference and then evaluates the generated tracking results automatically.
- `-M test` always creates a submission ZIP; when every selected test sequence has GT, it also runs evaluation automatically.

For resume training, checkpoint inheritance, distributed options, and parameter sweeps, see [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md).

## 6. DanceTrack Visualization

The offline tools use the DanceTrack main config. Replace all path placeholders before running them.

### Existing TXT Results

```bash
python script/offline_txt_eval_visualize.py \
  -C configs/dancetrack_train.yaml -M val -DR <Path to your dataset root> \
  -T <Path to tracking results> -V dancetrack0041,dancetrack0043
```

### Multi-Tracker Overlay

```bash
python script/offline_multi_txt_overlay_visualize.py \
  -C configs/dancetrack_train.yaml -DR <Path to your dataset root> -V dancetrack0007 \
  --tracker_dirs <Path to tracker A> <Path to tracker B> <Path to tracker C> \
  --tracker_labels Generative Learnable TAQ-MOTR \
  --output_dir outputs/results/DanceTrack/multi_overlay --focus_gt_id 4 --max_frames 30
```

### Query-Focus Visualization

```bash
python script/offline_query_focus_visualize.py \
  -C configs/dancetrack_train.yaml -DR <Path to your dataset root> -V dancetrack0007 \
  --method_ckpt_dirs <Path to generative checkpoint> <Path to learnable checkpoint> <Path to TAQ-MOTR checkpoint> \
  --method_labels Generative Learnable TAQ-MOTR \
  --output_dir outputs/results/DanceTrack --compose_header_mode compose \
  --crop_scope video --crop_margin 50 --exclude_original
```

## 7. Q&A

**Q: Where should `-DR` point?**

A: Use the common dataset root. For BFT, pass the parent of `BFT/`; for SportsMOT, do not pass the nested `SportMOT/dataset` directory because the code derives it automatically.

**Q: Why are model keys missing during evaluation?**

A: Use the matching main config. For a custom architecture, keep its original `<checkpoint_root>/train/config.yaml` beside the checkpoint.

**Q: Why does a distributed run hang?**

A: The number of GPU ids passed to `--gpus` must match `--nproc_per_node`.

See [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md) for detailed runtime and inference options.
