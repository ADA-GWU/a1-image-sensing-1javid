[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/GwLNboYD)

# Stereo Matching with Census Transform

Computes depth maps from stereo image pairs using census transform and cost aggregation.

## Folder Structure

```
images/
├── originals/
│   ├── dataset1/      (left.jpeg, right.jpeg)
│   ├── dataset2/      (left.jpeg, right.jpeg)
│   └── dataset3/      (left.jpeg, right.jpeg)
└── outputs/
    ├── dataset1/      (disparity maps)
    ├── dataset2/      (disparity maps)
    └── dataset3/      (disparity maps)
```

## Dataset Information

Camera settings for each dataset:

| Dataset | ISO | Focal Length | Aperture | Shutter Speed | Lighting Conditions |
|---------|-----|--------------|----------|---------------|---------------------|
| dataset1 | 500 | 48mm | f/1.78 | 1/40s | Indoor, artificial |
| dataset2 | 400 | 24mm | f/1.78 | 1/50s | Outdoor, 1 hour before sunset |
| dataset3 | 400 | 24mm | f/1.78 | 1/50s | Outdoor, 1 hour before sunset |

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `opencv-python` - Image processing and I/O
- `numpy` - Numerical operations and arrays
- `matplotlib` - Visualization and plotting
- `joblib` - Parallel processing for cost volume computation

## Usage

Run with default parameters:
```bash
python code.py
```

Or customize via command line:
```bash
python code.py --dataset dataset2 --window 90 --max-disp 192 --agg-window 50
```

Skip visualization:
```bash
python code.py --no-viz
```

See all options:
```bash
python code.py --help
```

Output files are named: `{dataset}-{window}-{disp}-{agg}.png`

---

## Results Analysis and Comparison

### Parameter Configurations Tested

**Dataset 1 (Indoor Books)**
- `dataset1-60-64-25`: Small window (60), low disparity (64), small aggregation (25)
- `dataset1-80-128-50`: Medium window (80), medium disparity (128), medium aggregation (50)
- `dataset1-120-164-75`: Large window (120), high disparity (164), large aggregation (75)

**Dataset 2 (Outdoor Trees)**
- `dataset2-70-164-25`: Small-medium window (70), high disparity (164), small aggregation (25)
- `dataset2-90-192-50`: Medium window (90), high disparity (192), medium aggregation (50)
- `dataset2-120-232-75`: Large window (120), very high disparity (232), large aggregation (75)

**Dataset 3 (Outdoor Statue)**
- `dataset3-60-120-25`: Small window (60), medium disparity (120), small aggregation (25)
- `dataset3-80-232-25`: Medium window (80), very high disparity (232), small aggregation (25)
- `dataset3-120-268-25`: Large window (120), maximum disparity (268), small aggregation (25)

---

### Dataset 1: Indoor Books Scene

**Scene Characteristics:**
- Highly textured surfaces (book spines with text and patterns)
- Objects at relatively uniform depth (all books on same shelf)
- High contrast between light and dark book covers
- Sharp edges and clear details

**Observations:**
- **Best performing dataset** due to rich texture and controlled lighting
- Clear depth separation visible between background shelf, individual book spines, and front edges of books
- Census transform works exceptionally well on textured surfaces
- Vertical structures (book spines) create distinct patterns
- Minimal noise and artifacts in the disparity map

**Best Performing Parameters:** `dataset1-80-128-50`

---

### Dataset 2: Outdoor Trees Scene

**Scene Characteristics:**
- Natural outdoor scene with complex organic structures
- Trees with thin branches and leaves create challenging textures
- Multiple depth layers: foreground bushes, mid-ground trees, background sky
- Lower contrast compared to indoor scene

**Observations:**
- **More challenging** than Dataset 1 due to repetitive textures in foliage, thin branches, and sky regions with minimal texture
- Noisy disparity in sky regions (low texture)
- Good performance especially on tree trunks (high texture)

**Challenges:** Texture-less sky, thin branches, repetitive patterns

**Best Performing Parameters:** `dataset2-120-232-75`

---

### Dataset 3: Outdoor Statue Scene

**Scene Characteristics:**
- Urban scene with geometric architectural elements
- Large statue as foreground subject
- Building facades with regular window patterns
- Ground surface with distinct textures (pavement, grass)
- Mix of smooth and textured surfaces

**Observations:**
- **Well-defined foreground subject**: Statue clearly separated from background
- Building windows create interesting texture patterns
- Some noise on smooth building walls (low texture)

**Challenges:** Smooth building walls, repetitive windows, sky regions with minimal texture

**Best Performing Parameters:** `dataset3-80-232-25`

---

### Quality Metrics

| Dataset | Overall Quality | Edge Preservation | Noise Level | Depth Clarity |
|---------|----------------|-------------------|-------------|---------------|
| Dataset 1 | Excellent | Very High | Very Low | Very Clear |
| Dataset 2 | Moderate | Medium | Medium-High | Moderate |
| Dataset 3 | Good | High | Low-Medium | Clear |

---

### Conclusion

The census transform-based stereo matching algorithm performs well across diverse conditions:

- **Best performance**: Textured, controlled lighting environments (Dataset 1)
- **Good robustness**: Handles outdoor natural lighting effectively (Datasets 2 & 3)
- **Parameter sensitivity**: Proper tuning of window sizes and disparity range is crucial