# Table Tennis Stroke Analysis

We introduce a novel method for table tennis strokes analysis.

## 三維姿態估測推論 3D Pose Estimation Inference

- **out_video_sf** : Start frame of input video. <br>
- **out_video_dl** : Output video length. <br>
- **pose3d_rotation** : z_rotate y_rotate x_rotate <br>

```bash
python common/pose3d/vis_longframes.py --video other_f1_left.mp4 --out_video_sf 0 --out_video_dl 1000 --pose3d_rotation 0 0 0
```

## 動作分析 Motion Analysis

- **subject1** : Subject 1. <br>
- **subject2** : Subject 2. <br>
- **mode** : analysis or benchmark <br>

**Analysis mode:**

```bash
python run.py --subject1 nchu_m7_left --subject2 other_f1_left --mode analysis
( python run.py --subject1 nchu_m7_left --subject2 other_f3_left --mode analysis )
```

**Benchmark mode:**

```bash
python run.py --subject1 nchu_m7_left --mode benchmark
```
