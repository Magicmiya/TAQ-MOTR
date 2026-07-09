# TAQ_MOTR

## 📖 Overview
Tracking-by-propagation (TBP) methods perform end-to-end multi-object tracking by propagating track queries across frames, jointly handling detection and identity association without heuristic post-processing while naturally modeling temporal information. However, as detection and association are tightly coupled, detection queries in TBP are expected not only to discover newborn objects but also, in certain cases, to complement or recover challenging trajectories with current-frame evidence. Existing methods largely inherit detection-query formation from single-frame detection and construct such queries before the decoder with little awareness of trajectory states. Consequently, newborn objects may be missed, while challenging tracked objects may lack complementary evidence, causing association failures or duplicate responses. To address these issues, we propose TAQ-MOTR, a tracking-aware detection query formation framework that employs a pre-decoder Hybrid Query Generator to fuse current-frame image features, track memory, and learnable memory. By conditioning detection queries on trajectory states, TAQ-MOTR adapts them to newborn-object discovery and tracked-object complementation. We further introduce a training-only auxiliary branch that decouples tracked-object complementation from newborn discovery and provides dedicated detection supervision for the corresponding detection queries. TAQ-MOTR improves performance on DanceTrack and SportsMOT, achieving 72.0 HOTA, 63.1 AssA, and 76.6 IDF1 on the DanceTrack test set.

![Overall structure](assets/Overall_structure.png)
![demo_focus](https://github.com/Magicmiya/TAQ-MOTR/blob/develop/assets/demo_focus.mp4)
<video src="assets/demo_focus.mp4" controls="controls" width="960" height="540"></video>


## 📣 News

- **2026.06.18**: We have built the full project and provided richer video examples for the model attention visualization. The HQG module code is now public, and the complete version will be fully released after the paper review stage is complete.

## 👀 Performance

### DanceTrack
  | Methods | HOTA $\uparrow$ | DetA $\uparrow$ | AssA $\uparrow$ | MOTA $\uparrow$ | IDF1 $\uparrow$ | checkpoint |
  | ------- | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :--------: |
  |TAQ-MOTR | 72.0 | 82.4|63.1|91.7|76.6| [Google](https://drive.google.com/file/d/1o6AStF-OoG7edE7TBYFVVTZnxbvUde32/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/1J-4PKw4vKr28FcrYlxUbJQ?pwd=ajpd)|

### SportsMOT
  | Methods | HOTA $\uparrow$ | DetA $\uparrow$ | AssA $\uparrow$ | MOTA $\uparrow$ | IDF1 $\uparrow$ | checkpoint |
  | ------- | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :--------: |
  |TAQ-MOTR | 72.0 | 82.4|63.1|91.7|76.6| [Google](https://drive.google.com/file/d/16OcPjhkYBHwov7IPZEIjZ2hM2nUoy9Zj/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/1WVFPfYvUCl2pYZUBA2xxMQ?pwd=zy9s)|

## 🔧 Quick Start

- See **[INSTALL.md](doc/INSTALL.md)** for instructions on installing the required components.
- See **[QUICK_START.md](doc/QUICK_START.md)** for quick-start instructions on model training, inference, and evaluation.

## :tada: Acknowledgements

This project is based on [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR), [DAB-DETR](https://github.com/IDEA-Research/DAB-DETR), [RT-DETRv2](https://github.com/zheli-hub/RT-DETRv2), and [TrackEval](https://github.com/JonathonLuiten/TrackEval). We thank the contributors of these excellent codebases.

## Citation
```bash
    The paper is still under review......
```
