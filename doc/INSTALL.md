# TAQ-MOTR Install
- Our codebase is built upon **Python 3.10.18** and **pytorch 2.3.1**. 
- To use the CUDA version of deformable attention, you'll need **CUDA Toolkit 11.8** as well. We do offer a **PyTorch-only** version that runs out of the box, but don't expect much in terms of speed—it's painfully slow :smirk:.
- 
## Environment Setup(Conda based)
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
- Quick import check:
```bash
python -c "import torch; import MultiScaleDeformableAttention as m; print(m.__file__)"
```

- Common build errors:
  - `which g++` failed: install `gcc_linux-64` and `gxx_linux-64`.
  - `cuda_runtime.h` missing: install `cuda-cudart-dev=11.8`.
  - `thrust/complex.h` missing or CUDA>=12 error from CCCL: pin `cuda-cccl=11.8`.


## Dataset Prepare
You can use scripts in `data/tools` to generate dataset-side auxiliary files.

`<path to your datasets root>` should be the parent directory that contains dataset folders, for example:

```text
<path to your datasets root>
|- DanceTrack/
|  |- train/
|  |- val/
|  `- test/
|- MOT17/
|  |- annotations/
|  |- train/
|  |- test/
`- SportMOT/
|  |- dataset/
   |    |- train/
   |    |- test/
```
```shell
# for DanceTrack
python data/tools/gen_dancetrack.py -P <path to your datasets root>
# for MOT17
python data/tools/gen_motchallenge_gts.py -P <path to your datasets root> -B MOT17
# for MOT20
python data/tools/gen_motchallenge_gts.py -P <path to your datasets root> -B MOT20
# optional: build COCO-style annotations for aux detection datasets
python data/tools/gen_aux_det_coco.py -P <path to your datasets root> -D CityPersons
python data/tools/gen_aux_det_coco.py -P <path to your datasets root> -D CrowdHuman
```
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