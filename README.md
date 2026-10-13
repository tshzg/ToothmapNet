# ToothMap FDI ONNX Inference Package / FDI ONNX 推理包

[English](#english) | [中文](#中文)

## English

This package includes an ONNX model, standalone inference code, and a dependency list. PyTorch, timm, and model architecture source code are not required.

**Model download:** The ONNX model file is available in this repository's **GitHub Releases**. Download `toothmap_fdi.onnx` from the release assets and place it in the same directory as `inference.py`, or specify its location with `--model`.

### License

Copyright (c) 2026 ToothmapNet Contributors. The original ONNX model (including embedded parameters), inference code, and project documentation are offered under the [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0). See [LICENSE.txt](LICENSE.txt) for the full terms, [license](license) for the scope notice, and [NOTICE.txt](NOTICE.txt) for copyright and redistribution notices.

Uses permitted by the license are allowed subject to its terms. Commercial uses outside those permissions require a separate written license from the copyright holder. Consult the full license for its definitions of permitted purposes and provisions for noncommercial organizations. Third-party dependencies retain their own licenses.

Training data and training recipes are not included. This distribution adds no restriction on modification or fine-tuning permitted by the license and does not add a no-training clause.

### Installation and usage

Validated with Python 3.12. A separate virtual environment is recommended.

```powershell
python -m pip install -r requirements.txt
python inference.py --input "path/to/image.png" --output predictions --device cpu
python inference.py --input "path/to/images" --output predictions
```

For GPU inference, install `onnxruntime-gpu==1.23.2` instead of `onnxruntime` (do not install both packages in the same environment), configure compatible CUDA/cuDNN runtime libraries, and use `--device cuda`. The default `--device auto` attempts an available CUDA provider and otherwise uses CPU.

Supported formats: PNG, JPG, JPEG, BMP, TIF, and TIFF. Directory input scans only the first level. Input files must have unique filename stems. Supply image files rather than JSON annotations. The default model path is `toothmap_fdi.onnx` beside the script; use `--model` to specify another path.

The default presence threshold is **0.55** (`--conf`). Presence uses only presence probability, without a heatmap-peak gate. No TTA, parabolic peak refinement, or channel conflict handling is performed. The default decoder bilinearly upsamples **logits** from 32×64 to 256×512 (half-pixel coordinates, equivalent to `align_corners=False`), applies spatial softmax at temperature 1, and computes the expected x/y coordinates. Do not apply sigmoid to heatmaps before softmax. Use `--decoder hardargmax` to compare native 32×64 peak decoding.

### Outputs

- `images_fdi`: visualizations labeled with standard FDI tooth numbers.
- `json_fdi`: Labelme JSON files containing only teeth predicted as present.
- `details`: presence probabilities and coordinates for all 32 FDI tooth positions.

All exposed tooth numbers use FDI notation. JSON `points` and the `x`/`y` fields in `details` use original-image pixel coordinates. `x_lb`/`y_lb` use letterbox coordinates (height 512, width 1024). Coordinates are `null` for teeth predicted as absent.

### ONNX interface

Input `image`: `float32 [1,3,512,1024]`. Convert the image to grayscale and then three identical channels, resize with preserved aspect ratio and centered zero padding, divide by 255, and normalize with `mean=[0.485,0.456,0.406]` and `std=[0.229,0.224,0.225]`.

Output channels follow standard FDI order: **11–18, 21–28, 31–38, 41–48**. Both outputs contain logits. Apply sigmoid to presence logits; the localization decoder applies spatial softmax to upsampled heatmap logits.

| Output name | Shape | Description |
|---|---|---|
| `heatmap_logits_fdi` | `[1,32,32,64]` | Heatmaps without TTA |
| `presence_logits_fdi` | `[1,32]` | Presence scores without TTA |

FDI ordering is embedded in the model. The ONNX graph executes one forward pass with no flipped-image branch. Input dimensions and batch size (1) are fixed. The script processes images individually. Coordinates scale to letterbox using x / (512−1) × (1024−1) and y / (256−1) × (512−1), then map to the original image.

The package does not include training configurations, training code, training data, or patient images. ONNX is a deployment format, not an encryption format.

## 中文

本包提供 ONNX 模型、独立推理代码及依赖。无需 PyTorch、timm 或网络结构源码。

**模型下载：** ONNX 模型文件位于本仓库的 **GitHub Releases** 页面。请从 Release 附件中下载 `toothmap_fdi.onnx`，放到 `inference.py` 同目录，也可通过 `--model` 指定模型路径。

### 许可

Copyright (c) 2026 ToothmapNet Contributors。本包的原始 ONNX 模型（含嵌入参数）、推理代码及项目文档采用 [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0)。完整条款见 [LICENSE.txt](LICENSE.txt)，范围说明见 [license](license)，版权及分发声明见 [NOTICE.txt](NOTICE.txt)。

允许条款所规定的非商业用途；超出许可证授权范围的商业用途需向版权持有人取得单独书面许可。具体允许的用途及非商业组织规定以完整许可证为准。第三方依赖仍采用各自许可证。

包内未提供训练数据和训练配方；本许可证不额外禁止其允许范围内的修改或微调。许可证原文未添加禁止训练条款。

### 安装和使用

验证环境为 Python 3.12。建议使用独立虚拟环境：

```powershell
python -m pip install -r requirements.txt
python inference.py --input "path/to/image.png" --output predictions --device cpu
python inference.py --input "path/to/images" --output predictions
```

如需 GPU，使用 onnxruntime-gpu==1.23.2 替代 onnxruntime（不要同时安装两种包），并配置匹配的 CUDA/cuDNN 运行库，再使用 --device cuda。auto 优先尝试可用的 CUDA，否则使用 CPU。

支持 PNG/JPG/JPEG/BMP/TIF/TIFF。目录只扫描第一层，不同输入文件须具有不同的文件名主体。输入为图像，不是 JSON。模型默认读取脚本同目录的 toothmap_fdi.onnx，也可用 --model 指定。

默认存在性阈值为 **0.55**（`--conf`），仅使用 presence 概率，不加热图峰值门槛。不进行 TTA、抛物线峰值细化或通道冲突处理。默认先将 32×64 的 **logits** 双线性放大到 256×512（半像素坐标，与 `align_corners=False` 一致），再以温度 1 做空间 softmax，计算 x/y 坐标期望；不是先对热图做 sigmoid。可用 `--decoder hardargmax` 对照原始 32×64 峰值取点。

### 输出

- images_fdi：标准 FDI 编号可视化。
- json_fdi：只包含预测存在牙位的 Labelme JSON。
- details：32 个标准 FDI 牙位的存在性概率与坐标。

所有对外编号均为 FDI。JSON 的 points、details 的 x/y 使用原图像素坐标；x_lb/y_lb 使用高 512 × 宽 1024 的 letterbox 坐标。未检出的牙位坐标为 null。

### ONNX 接口

输入 image：float32 [1,3,512,1024]，灰度图转三通道，letterbox 居中填充，除以 255 后按 mean=[0.485,0.456,0.406]、std=[0.229,0.224,0.225] 归一化。

输出通道统一为：**11–18、21–28、31–38、41–48**。两个输出均为 logits。存在性通过 sigmoid 转换为概率；定位对放大后的热图 logits 做空间 softmax：

| 输出名 | 形状 | 内容 |
|---|---|---|
| heatmap_logits_fdi | [1,32,32,64] | 普通热图 |
| presence_logits_fdi | [1,32] | 普通存在性 |

FDI 顺序已封装进模型。ONNX 仅执行一次前向计算，不含翻转图分支。输入尺寸和 batch=1 固定，脚本逐张处理。坐标按 x / (512−1) × (1024−1)、y / (256−1) × (512−1) 换算至 letterbox，再映射回原图。

包内不包含训练配置、训练代码、训练数据或患者影像。ONNX 是部署格式，并不是加密格式。
