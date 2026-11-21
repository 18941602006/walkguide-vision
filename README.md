# WalkGuide Vision

面向视障出行研究的本地视觉辅助原型，支持图片分割、室内外实时视频分析与视频导出。

## 功能

- 图片上传、语义分割叠图和类别掩码下载。
- 摄像头和本地视频实时分析，显示分割帧率及请求耗时。
- 室外近处路面类别选择、障碍识别和候选通道边界收紧。
- 室内地板、障碍、楼梯与台阶识别，可选低光增强。
- 导出与所选室内外模式一致的无声 WebM 视频。
- 在候选通道内生成白色方向线，显示向前、左右调整或暂停确认。红色终点表示需要暂停确认。

方向线使用当前画面的像素通道规划，优先选择离边缘较远的位置，并排除已识别的障碍。无法确认足够长、连续的通道时提示暂停。该线尚未结合定位、目的地、真实身体宽度或 ToF 距离，分割错误也会影响方向提示。

绿色表示视觉候选通道，橙色表示识别到的障碍，蓝色表示室内楼梯或台阶。当前算法仍存在漏检、误分割和分叉处外溢，颜色与边缘约束也可能使阴影、弯道处的候选区域缩小。该项目尚未接入 ToF 测距，真实高差、净宽、来车情况与安全通行性需要独立验证。

## 本地运行

建议使用 Python 3.11。使用 NVIDIA GPU 时，先按 PyTorch 官方说明安装适合显卡的 CUDA 版本。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python setup_models.py
python web_app.py
```

打开 [本地网页](http://127.0.0.1:7860/)。摄像头需要浏览器授权，服务默认监听本机地址。

选择「室外通道」或「室内走廊」，再打开摄像头或本地视频。服务优先处理最新帧，繁忙和超时请求会跳过。视频导出支持 100 MB 以内的 MP4、MOV、AVI 和 WebM，输出长边最多 960 像素并保留帧率。

## 模型和数据

| 用途 | 模型 | 获取方式 |
| --- | --- | --- |
| 图片 | WalkScapes 微调的 SegFormer-B0 | 仓库的 `model/` |
| 室外视频 | NVIDIA Cityscapes SegFormer-B2 | `python setup_models.py` |
| 室内视频 | NVIDIA ADE20K SegFormer-B2 | `python setup_models.py` |

公开视频权重从原发布方下载，使用时须遵守其许可。所使用的 NVIDIA SegFormer 权重及其派生模型限非商业研究或评估用途，许可原文保存在 `model/NVIDIA-LICENSE.txt`。[Cityscapes 模型](https://huggingface.co/nvidia/segformer-b2-finetuned-cityscapes-1024-1024) · [ADE20K 模型](https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512)

图片模型基于 WalkScapes Dataset，© 2026 Kacper Chabros 等，数据许可为 CC BY-NC 4.0。该模型由 ImageNet 预训练的 SegFormer-B0 微调，分割头训练于 WalkScapes，适用许可和署名信息应随使用保留。训练与评估指标位于 `model/metrics.json`。[原数据、许可与条款](https://github.com/KNSyNAPS-org/walkscapes-dataset-data) · [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) · [原编码器及其许可](https://huggingface.co/nvidia/mit-b0)

## 图片推理与训练

```powershell
python walkscapes.py predict photo.jpg --mask-output mask.png
python download.py
hf download nvidia/mit-b0 config.json pytorch_model.bin --local-dir data/mit-b0
python walkscapes.py train --epochs 10
```

默认训练数据保存在 `data/walkscapes-dataset-data/dataset`，保留官方训练、验证和测试划分。自定义路径可传入 `--data` 或 `--encoder`。命令行图片模型采用 14 类语义标签，ID 255 在训练和评估时忽略。

## 发布内容

仓库包含源码、依赖说明、模型下载脚本及公开数据训练的图片模型。个人视频、截图、测试对比图、日志、环境文件、凭据和本机绝对路径均排除。新增素材在提交前仍应人工检查。
