# ToothMap FDI ONNX 推理包

本包提供 ONNX 模型、独立推理代码及依赖。无需 PyTorch、timm 或网络结构源码。

## 许可

Copyright (c) 2026 Shaohua Tang。本包的原始 ONNX 模型（含嵌入参数）、推理代码及项目文档采用 [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0)。完整条款见 [LICENSE.txt](LICENSE.txt)，范围说明见 [license](license)，版权及分发声明见 [NOTICE.txt](NOTICE.txt)。

允许条款所规定的非商业用途；超出许可证授权范围的商业用途需向版权持有人取得单独书面许可。具体允许的用途及非商业组织规定以完整许可证为准。第三方依赖仍采用各自许可证。

包内未提供训练数据和训练配方；本许可证不额外禁止其允许范围内的修改或微调。许可证原文未添加禁止训练条款。

## 安装和使用

验证环境为 Python 3.12。建议使用独立虚拟环境：

```powershell
python -m pip install -r requirements.txt
python inference.py --input "path/to/image.png" --output predictions --device cpu
python inference.py --input "path/to/images" --output predictions
```

如需 GPU，使用 onnxruntime-gpu==1.23.2 替代 onnxruntime（不要同时安装两种包），并配置匹配的 CUDA/cuDNN 运行库，再使用 --device cuda。auto 优先尝试可用的 CUDA，否则使用 CPU。

支持 PNG/JPG/JPEG/BMP/TIF/TIFF。目录只扫描第一层，不同输入文件须具有不同的文件名主体。输入为图像，不是 JSON。模型默认读取脚本同目录的 toothmap_fdi.onnx，也可用 --model 指定。

默认启用亚像素细化和水平翻转 TTA；--no-refine、--no-tta 可分别关闭。--conf 设置存在性概率阈值（默认 0.30）；另要求热图峰值至少 0.30。--min-sep 为通道冲突处理距离阈值（默认 10 个 letterbox 像素）。

## 输出

- images_fdi：标准 FDI 编号可视化。
- json_fdi：只包含预测存在牙位的 Labelme JSON。
- details：32 个标准 FDI 牙位的存在性概率与坐标。

所有对外编号均为 FDI。JSON 的 points、details 的 x/y 使用原图像素坐标；x_lb/y_lb 使用高 512 × 宽 1024 的 letterbox 坐标。未检出的牙位坐标为 null。

## ONNX 接口

输入 image：float32 [1,3,512,1024]，灰度图转三通道，letterbox 居中填充，除以 255 后按 mean=[0.485,0.456,0.406]、std=[0.229,0.224,0.225] 归一化。

输出通道统一为：11–18、21–28、31–38、41–48。四个输出均为 logits，使用前需 sigmoid：

| 输出名 | 形状 | 内容 |
|---|---|---|
| heatmap_logits_fdi | [1,32,32,64] | 普通热图 |
| presence_logits_fdi | [1,32] | 普通存在性 |
| heatmap_logits_fdi_tta | [1,32,32,64] | 水平翻转 TTA 融合热图 |
| presence_logits_fdi_tta | [1,32] | 水平翻转 TTA 融合存在性 |

FDI 输出顺序及 TTA 对齐均已封装进模型。TTA 使用原图与翻转图的 logits 均值。模型采用固定输入尺寸和 batch=1；推理代码逐张处理。只请求普通输出时，ONNX Runtime 是否裁剪未请求分支取决于运行时优化，不能据此保证减少一半计算量。

包内不包含训练配置、训练代码、训练数据或患者影像。ONNX 是部署格式，并不是加密格式。
