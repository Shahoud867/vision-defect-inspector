# FAST National University of Computer and Emerging Sciences
## Department of Artificial Intelligence and Data Science
## CS4059: Fundamentals of Computer Vision
## Fall 2026
## Assignment 1

### Industrial Defect Inspection & Surface Crack Detection via Custom Linear Filtering and Edge Processing

**Instructions:**

- **Assigned Date:** 31 August 2026
- **Due Date:** 14 September 2026 (23:59 PM)
- **Work Policy:** Strictly Individual Assignment. Plagiarism and direct sharing of code will lead to immediate referral to the disciplinary committee with an automatic zero grade.
- **Allowed Libraries:** numpy, matplotlib, scipy.io (for data loading), and cv2 / PIL.
- **Strict Prohibition:** cv2 / PIL may only be used for basic image I/O (cv2.imread, cv2.imwrite) and color space conversion (cv2.cvtColor). Any use of built-in processing functions (e.g., cv2.filter2D, cv2.GaussianBlur, cv2.Sobel, cv2.Canny) is strictly forbidden and results in zero marks for that respective task.

---

## 1 Problem Scenario

In high-precision manufacturing (solar cells, silicon wafers, aerospace turbine blades) and document forensics, visual surface inspection is a critical first-line automated defense. Sensor imagery captured in real-world facilities is inherently degraded by shot noise, motion blur, and non-uniform illumination.

In this assignment, you will construct an industrial-grade **Visual Quality Inspection & Micro-Crack Profiling Pipeline** in Python from mathematical first principles. You will implement spatial 2D and 3D convolutions, Gaussian noise attenuation, high-pass unsharp masking, multi-channel pooling, and an exact 4-stage Canny edge detector from scratch using pure vectorized NumPy.

### 1.1 Evaluation Datasets

You must benchmark and report your implementation on sample images selected from the following benchmark repositories:

1. **PCB (Printed Circuit Board) Defect Dataset:** High-contrast circuit board images with manufacturing flaws (mouse bites, spurs, shorts, open circuits).
   Link: https://www.kaggle.com/datasets/akhatova/pcb-defects

2. **Concrete Surface Crack Detection Dataset:** Concrete structural surfaces under diverse lighting, containing fine micro-cracks alongside textured backgrounds.
   Link: https://www.kaggle.com/datasets/arunrk7/surface-crack-detection

### 1.2 General Assumptions & Implementation Hints

- **Grayscale Conversion:** The Canny pipeline operates on single-channel images. Ensure you convert RGB dataset images to grayscale before processing.
- **Kernel Sizes:** For padding calculations, you may assume that all spatial filters (Gaussian, Sobel, Custom) will have odd dimensions (e.g., 3 × 3, 5 × 5).
- **Cross-Correlation vs. Convolution:** For Task 1, implementing spatial cross-correlation (sliding window dot product without flipping the kernel) is perfectly acceptable and standard practice in modern CV.
- **Normalization:** Remember that gradient computations will yield negative numbers, and magnitudes may exceed 255. Normalize your arrays to [0, 255] and cast to uint8 before saving/displaying to avoid visual artifacts.

---

## 2 Detailed Task Specifications

### 2.1 Task 1: Generalized 2D and 3D Convolution Engines (20 Marks)

1. **2D Spatial Convolution Function [10 Marks]:**

   Implement `conv2d(image, kernel, stride=1, padding='same', pad_value=0)`.

   - Support 'same' (output dimensions match input when S = 1) and 'valid' (no padding).
   - Output dimensions must strictly satisfy the discrete spatial formula:

$$W_{out} = \left\lfloor \frac{W_{in} - F + 2P}{S} \right\rfloor + 1, \quad H_{out} = \left\lfloor \frac{H_{in} - F + 2P}{S} \right\rfloor + 1$$

   where $W_{in}, H_{in}$ are input dimensions, $F$ is filter size, $P$ is pad width, and $S$ is stride.

2. **3D Multi-Channel Convolution Layer [10 Marks]:**

   Implement `conv3d(image, kernel_stack, stride=1, padding='same')` where `image` has shape (H, W, C) and `kernel_stack` contains K kernels of shape (kh, kw, C).

   - For each filter, compute channel-wise element multiplication, sum across all C channels, and stack into an activation volume (H_out, W_out, K).

### 2.2 Task 2: Filtering, Contrast Sharpening & Subsampling (20 Marks)

1. **Isotropic 2D Gaussian Kernel [6 Marks]:**

   Implement `generate_gaussian_kernel_2d(ksize, sigma)` where:

$$G_\sigma(x, y) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$$

   Hint: Use `np.meshgrid` to generate the (x, y) coordinate grid. Normalize the final kernel such that $\sum_{x,y} G_\sigma(x, y) = 1$.

2. **Unsharp Masking (High-Pass Detail Accentuation) [7 Marks]:**

   Implement `unsharp_mask(image, sigma=1.5, alpha=1.2)` using detail amplification:

$$F_{sharp} = F + \alpha \cdot (F - F * G_\sigma)$$

   Clip output values strictly to [0, 255] to prevent overflow artifacts.

3. **Feature Subsampling via 2D Pooling [7 Marks]:**

   Implement `pool2d(feature_map, pool_size=(2, 2), stride=2, mode='max')` supporting both 'max' and 'average' modes.

   - **Theoretical Reflection:** In your report, write down the analytical gradient matrix $\frac{\partial Output}{\partial Input}$ for a 2 × 2 max pooling block with input $\begin{bmatrix} 8 & 7 \\ 12 & 9 \end{bmatrix}$ and explain why gradients route exclusively to the maximum activation.

### 2.3 Task 3: First-Principles Canny Edge Detection Pipeline (30 Marks)

Implement the complete 4-stage edge detection pipeline:

1. **Stage 1: Pre-smoothing & Sobel Gradient Computation [8 Marks]:**

   Convolve the grayscale image with horizontal and vertical normalized Sobel kernels:

$$S_x = \frac{1}{8}\begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \quad S_y = \frac{1}{8}\begin{bmatrix} 1 & 2 & 1 \\ 0 & 0 & 0 \\ -1 & -2 & -1 \end{bmatrix}$$

   Compute magnitude $M = \sqrt{S_x^2 + S_y^2}$ and direction $\theta = \text{atan2}(S_y, S_x)$.

2. **Stage 2: Non-Maximum Suppression (NMS) [12 Marks]:**

   Convert θ to degrees in [0, 180°) and quantize into 4 sectors:

   - 0° ± 22.5° or 180° ± 22.5°: Compare with horizontal neighbors (i, j − 1) and (i, j + 1).
   - 45° ± 22.5°: Compare with diagonal neighbors (i − 1, j + 1) and (i + 1, j − 1).
   - 90° ± 22.5°: Compare with vertical neighbors (i − 1, j) and (i + 1, j).
   - 135° ± 22.5°: Compare with anti-diagonal neighbors (i − 1, j − 1) and (i + 1, j + 1).

   Set pixel to 0 if its magnitude is not strictly greater than both directional neighbors. Set image borders to 0.

3. **Stage 3: Double Thresholding & 8-Connected Hysteresis [10 Marks]:**

   Categorize pixels into Strong ($M \geq T_{high}$), Weak ($T_{low} \leq M < T_{high}$), and Non-edge ($M < T_{low}$). Retain weak pixels if and only if they connect directly or transitively to a strong edge via 8-neighborhood connectivity.

### 2.4 Task 4: Step-by-Step Visual Ablation Study (15 Marks)

To systematically inspect the pipeline, implement and document an end-to-end ablation study:

1. **Ablation 1 (Impact of Noise & Smoothing):** Synthesize noisy test samples by adding Gaussian noise $\mathcal{N}(0, \sigma_n^2)$ with $\sigma_n \in \{10, 25, 40\}$. Run gradient computation directly without pre-smoothing vs. with Gaussian pre-smoothing. Display the resultant gradient noise maps.

2. **Ablation 2 (Impact of NMS):** Show edge maps before NMS (blurred, thick gradient bands) vs. immediately after NMS (1-pixel wide thin ridges).

3. **Ablation 3 (Impact of Hysteresis vs. Single Thresholding):** Compare standard double-threshold hysteresis edge linking against a single global threshold. Highlight how hysteresis preserves continuous weak crack lines without introducing excessive background texture noise.

### 2.5 Task 5: Scale-Space, Sensitivity & Failure Case Analysis (15 Marks)

1. **Scale-Space Crack Merging (σ ∈ {1.0, 2.5, 5.0}) [5 Marks]:** Evaluate crack detection across scale. Verify and document Witkin's scale-space causality: why edges shift or merge at larger σ but new edges never emerge.

2. **Parameter Sensitivity Grid [5 Marks]:** Plot a 3×3 grid varying $(T_{high}, T_{low})$ across noisy concrete and PCB samples to demonstrate optimal operating regimes.

3. **Failure Case Investigation [5 Marks]:** Identify and display at least two realistic scenarios where Canny edge detection fails (e.g., strong shadow contours, highly textured asphalt/concrete surfaces). Explain the theoretical limitations causing these failures.

---

## 3 Starter Implementation Skeleton

```python
import numpy as np
import cv2  # ONLY used for cv2.imread, cv2.imwrite, cv2.cvtColor
from collections import deque  # Recommended for efficient O(1) pop operations


def conv2d(
    image: np.ndarray, kernel: np.ndarray, stride: int = 1, padding: str = "same"
) -> np.ndarray:
    """
    Pure 2D spatial convolution from scratch.
    Assumes odd-sized kernels for 'same' padding logic.
    """
    H, W = image.shape
    k_h, k_w = kernel.shape

    if padding.lower() == "same":
        pad_h = (k_h - 1) // 2
        pad_w = (k_w - 1) // 2
        padded_img = np.pad(
            image, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=0
        )
    elif padding.lower() == "valid":
        padded_img = image
    else:
        raise ValueError("Padding must be 'same' or 'valid'")

    out_h = (padded_img.shape[0] - k_h) // stride + 1
    out_w = (padded_img.shape[1] - k_w) // stride + 1
    output = np.zeros((out_h, out_w), dtype=np.float64)

    for i in range(out_h):
        for j in range(out_w):
            r_start = i * stride
            c_start = j * stride
            region = padded_img[r_start : r_start + k_h, c_start : c_start + k_w]
            output[i, j] = np.sum(region * kernel)

    return output


def hysteresis_tracking(img_nms: np.ndarray, low_thresh: float, high_thresh: float) -> np.ndarray:
    """
    Double thresholding and 8-connectivity edge linking via BFS.
    """
    H, W = img_nms.shape
    strong_i, strong_j = np.where(img_nms >= high_thresh)

    output = np.zeros((H, W), dtype=np.uint8)
    output[strong_i, strong_j] = 255

    # Breadth-first edge tracking using an efficient deque
    queue = deque(zip(strong_i, strong_j))
    while queue:
        ci, cj = queue.popleft()
        for ni in range(max(0, ci - 1), min(H, ci + 2)):
            for nj in range(max(0, cj - 1), min(W, cj + 2)):
                if (img_nms[ni, nj] >= low_thresh) and (output[ni, nj] == 0):
                    output[ni, nj] = 255
                    queue.append((ni, nj))
    return output
```

---

## 4 Evaluation Rubric & Marking Scheme

| Task Component | Evaluation Criteria | Marks |
|---|---|---|
| Task 1: Convolution Engine | Correct conv2d and conv3d implementations; correct stride/padding dimensions; zero built-in filtering calls. | 20 |
| Task 2: Enhancement & Pooling | Mathematical Gaussian kernel synthesis; unsharp mask contrast sharpening; max and average pooling with analytical derivative derivations. | 20 |
| Task 3: Canny Edge Pipeline | Sobel operator, accurate 4-sector NMS, robust 8-connected hysteresis edge linking without broken edge artifacts. | 30 |
| Task 4: Visual Ablations | Rigorous visual ablations (Noise → Smoothing, Pre/Post NMS, Hysteresis vs Single thresholding) with side-by-side comparative subplots. | 15 |
| Task 5: Scale-Space & Analysis | Scale-space causality verification (σ = 1.0, 2.5, 5.0), (T_high, T_low) sensitivity matrix, documented failure case analysis on PCB/Concrete datasets. | 15 |
| **Total** | | **100** |

### Submission Package Requirements

Submit a single compressed file `<RollNumber>_Assignment01.zip` structured as follows:

- `src/`: All source scripts (.py or .ipynb) implementing each required module.
- `output_images/`: High-resolution output figures corresponding to each visual ablation and sensitivity experiment.
- `report.pdf`: Maximum 4-page report (IEEE or standard academic format) containing side-by-side visual ablation figures, quantitative parameter comparisons, failure case discussion, and theoretical answers.
