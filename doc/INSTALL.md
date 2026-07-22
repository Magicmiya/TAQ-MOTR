# TAQ-MOTR Installation

This project is built with **Python 3.10.18** and **PyTorch 2.3.1**. The CUDA version of deformable attention requires **CUDA Toolkit 11.8**. A PyTorch-only fallback is available, but it is intended only for debugging because it is much slower.

## Environment Setup

```bash
conda env create -f environment.yml
conda activate TAQ_MOTR
```

Notes:
- `environment.yml` already includes runtime dependencies and CUDA build toolchain required by `module/nn/ops/make.sh`.
- Keep CUDA build packages aligned with `pytorch-cuda=11.8`.

## Build Deformable Attention CUDA Ops

```bash
cd ./module/nn/ops
./make.sh
python test.py
```

Quick import check:

```bash
python -c "import torch; import MultiScaleDeformableAttention as m; print(m.__file__)"
```

Common build errors:
- `which g++` failed: install `gcc_linux-64` and `gxx_linux-64`.
- `cuda_runtime.h` missing: install `cuda-cudart-dev=11.8`.
- `thrust/complex.h` missing or a CUDA >= 12 error from CCCL: pin `cuda-cccl=11.8`.


## Dataset Prepare
You can use scripts in `data/tools` to generate dataset-side auxiliary files.

`<path to your datasets root>` should be the parent directory that contains dataset folders, for example:

```text
<path to your datasets root>
|- DanceTrack/
|  |- train/
|  |- val/
|  `- test/
|- BFT/
|  |- annotations_mot/
|  |- train/
|  |- val/
|  `- test/
|- MOT17/
|  |- annotations/
|  |- train/
|  |- test/
`- SportMOT/
|  `- dataset/
|     |- train/
|     |- val/
|     |- test/
|     `- ...
```

```shell
# for DanceTrack
python data/tools/gen_dancetrack.py -P <path to your datasets root>
# for BFT: generate the DanceTrack-compatible layout
python data/tools/gen_bft.py -P <path to your datasets root>
# for MOT17
python data/tools/gen_motchallenge_gts.py -P <path to your datasets root> -B MOT17
# for MOT20
python data/tools/gen_motchallenge_gts.py -P <path to your datasets root> -B MOT20
# optional: build COCO-style annotations for aux detection datasets
python data/tools/gen_aux_det_coco.py -P <path to your datasets root> -D CityPersons
python data/tools/gen_aux_det_coco.py -P <path to your datasets root> -D CrowdHuman
```

Expected generated files:

- DanceTrack: `DanceTrack/train_seqmap.txt`, `DanceTrack/val_seqmap.txt`, and `DanceTrack/test_seqmap.txt`.
- BFT: an eight-digit `img1/` view, normalized `gt/gt.txt`, and `seqinfo.ini` under every sequence in `BFT/{train,val,test}`. By default, `img1/` contains relative symbolic links to the original six-digit JPG files.
- MOT17: `MOT17/annotations/train.json`, `train_half.json`, `val_half.json`, `test.json`, plus `gt_train_half.txt` and `gt_val_half.txt` under each train sequence `gt/` folder. If sequence `det/det.txt` files exist, `det_train_half.txt` and `det_val_half.txt` are generated under `det/`.
- MOT20: `MOT20/annotations/train.json`, `train_half.json`, `val_half.json`, `test.json`, plus `gt_half-train.txt` and `gt_half-val.txt` under each train sequence `gt/` folder. If sequence `det/det.txt` files exist, `det_half-train.txt` and `det_half-val.txt` are generated under `det/`.
- CityPersons: `CityPersons/annotations/train.json`.
- CrowdHuman: `CrowdHuman/annotations/train.json` and `CrowdHuman/annotations/val.json`.

### BFT preprocessing

Pass the common dataset root to `-P`, not the nested `BFT` directory. For example, if the dataset is located at `/home/liu/Train/Data/BFT`, use:

```shell
python data/tools/gen_bft.py -P /home/liu/Train/Data
```

The preprocessing preserves the original images and `annotations_mot` files. For each sequence, it creates the layout consumed by the `BFT(DanceTrack)` dataset class:

```text
BFT/train/An3014/
|- 000001.jpg
|- 000002.jpg
|- img1/
|  |- 00000001.jpg -> ../000001.jpg
|  `- 00000002.jpg -> ../000002.jpg
|- gt/
|  `- gt.txt
`- seqinfo.ini
```

The BFT placeholders in `frame,id,x,y,w,h,-1,-1,-1,-1` are normalized to the valid single-class MOT ground-truth form `frame,id,x,y,w,h,1,1,1`. The script is repeatable: matching generated files are reported as `unchanged`.

Useful options:

- `--splits train`: preprocess only the selected split or splits, for example `--splits train val`.
- `--copy-images`: copy images into `img1/` instead of creating symbolic links.
- `--force`: replace generated files that exist with different content or link targets.

Optional aux detection layout:

```text
<path to your datasets root>
|- CityPersons/
|  |- annotations/train.json
|  |- citypersons.train
|  |- images/
|  `- labels_with_ids/
`- CrowdHuman/
   |- annotation_train.odgt
   |- annotation_val.odgt
   |- CrowdHuman_train01/Images/
   |- CrowdHuman_train02/Images/
   |- CrowdHuman_train03/Images/
   |- CrowdHuman_val/Images/
   `- annotations/train.json
```
