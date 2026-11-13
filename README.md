# OVSeg：复现与 RTX 4090 使用说明

本仓库包含完整官方模型源码，不新增数据转换或评测程序。**2080 Ti 官方示例推理已验证；4090 环境未验证；自有数据未验证。** 数据转换与注册由使用者完成。当前说明不能代替目标机器的实际验收。 

## 1. 克隆这一个仓库

```bash
mkdir -p /data/yukun/ov-seg
cd /data/yukun/ov-seg
git clone https://github.com/yiyipupu/ov-seg.git src
mkdir -p cache/{pip,conda,tmp,torch} checkpoints datasets outputs weights
cd src
export OVSEG_ROOT=/data/yukun/ov-seg
export PIP_CACHE_DIR=$OVSEG_ROOT/cache/pip
export CONDA_PKGS_DIRS=$OVSEG_ROOT/cache/conda
export TMPDIR=$OVSEG_ROOT/cache/tmp
export TORCH_HOME=$OVSEG_ROOT/cache/torch
export XDG_CACHE_HOME=$OVSEG_ROOT/cache
export DETECTRON2_DATASETS=$OVSEG_ROOT/datasets
```

每次新终端重新设置以上变量。环境、缓存及权重都在源码目录之外，不提交到 Git。

## 2. 配置 4090 环境（建议配置，尚未实机验证）

先运行 `nvidia-smi`、`nvcc --version`、`gcc --version`。建议 CUDA Toolkit 11.8、GCC 11，且 NVIDIA 驱动满足 CUDA 11.8 要求。`nvidia-smi` 显示的是驱动可支持的 CUDA 上限，不代表已安装 nvcc。没有 Toolkit 11.8 时先由管理员安装，或使用独立 Toolkit；不要覆盖其他项目的 CUDA。

[CUDA 11.8 支持 Ada](https://docs.nvidia.com/cuda/ada-compatibility-guide/)。选择 [PyTorch 官方配套版本](https://pytorch.org/get-started/previous-versions/) torch 2.0.1 + torchvision 0.15.2 / cu118，避免将本机 cu113 的 Detectron2 wheel 直接用于 4090。

```bash
conda create -p "$OVSEG_ROOT/env" python=3.8.20 pip=24.2 -y
conda activate "$OVSEG_ROOT/env"
python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-4090.txt
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS=4
git clone https://github.com/facebookresearch/detectron2.git "$OVSEG_ROOT/detectron2"
git -C "$OVSEG_ROOT/detectron2" checkout d1e04565d3bec8719335b88be9e9b961bf3ec464
python -m pip install --no-build-isolation -e "$OVSEG_ROOT/detectron2"
python -m pip install --no-build-isolation -e third_party/CLIP
python -m pip check
python -c 'import torch, torchvision, detectron2, clip, open_vocab_seg; print(torch.__version__, torchvision.__version__, detectron2.__version__); print(torch.cuda.get_device_name(0))'
python -c 'import torch; from detectron2.layers import nms; print(nms(torch.tensor([[0.,0.,10.,10.]],device="cuda"),torch.tensor([0.9],device="cuda"),0.5))'
```

Detectron2 固定官方 v0.6 commit，从源码针对 8.9 编译，不使用本机 cu113 wheel。[编译兼容说明](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md)。若编译或导入失败，保存日志继续排查；这里不声称该组合已验证。不要随意升级 torch 或修改评测算法。

### 本机 2080 Ti 的实际安装记录

Python 3.8.20、torch 1.10.1+cu113、torchvision 0.11.2+cu113、Detectron2 0.6 官方 cu113/torch1.10 wheel；NumPy 1.23.5、Pillow 9.5.0、timm 0.6.13。其余版本见 `requirements-local-2080ti.txt`（本机记录清单；CLIP 仍从本仓库安装，勿作为 4090 安装清单）。

本机创建独立 conda prefix 后的核心命令为：

```bash
python -m pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
python -m pip install -r requirements-4090.txt
python -m pip install detectron2==0.6 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
python -m pip install --no-build-isolation -e third_party/CLIP
```

依赖约束用于避免旧代码的 np.int/np.float 和 Pillow 接口兼容问题；无模型代码修改。上述 2080 Ti 命令不是 4090 默认配置。

## 3. 下载权重

```bash
curl -L --fail --retry 3 'https://drive.usercontent.google.com/download?id=1cn-ohxgXDrDfkzC1QdO-fi8IjbjXmgKy&export=download&confirm=t' -o "$OVSEG_ROOT/weights/ovseg_swinbase_vitL14_ft_mpt.pth"
curl -L --fail --retry 3 'https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt' -o "$OVSEG_ROOT/weights/ViT-L-14.pt"
echo '526080bf24e7182a8b5df4b92c5a14ec  '"$OVSEG_ROOT/weights/ovseg_swinbase_vitL14_ft_mpt.pth" | md5sum -c -
echo 'b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836  '"$OVSEG_ROOT/weights/ViT-L-14.pt" | sha256sum -c -
```

OVSeg 完整 MD5 来自本机下载文件，官方只给出短值 526080；CLIP SHA-256 来自官方文件 URL。Google 下载失败可在浏览器使用 [官方链接](https://drive.google.com/file/d/1cn-ohxgXDrDfkzC1QdO-fi8IjbjXmgKy/view)。不要将 HTML 提示页当作权重。

默认使用最大的 Swin-B + ViT-L/14，完整微调 + mask prompt tuning。官方另有 R101c + ViT-B/16 小模型和 mpt-only 消融权重，分别需要匹配配置；见 GETTING_STARTED.md。本流程无需另下载 Swin 预训练 pkl，命令行会覆盖 MODEL.WEIGHTS。

## 4. 自有数据应整理成什么格式

建议：

```text
/data/yukun/ov-seg/datasets/custom/
  images/test/000001.jpg
  labels/test/000001.png
```

每张图片对应同尺寸单通道类别 ID 标签；K 个类别用连续 ID 0..K-1，与 `stuff_classes` 的类别名称顺序完全一致。K≤255 时可用 uint8 PNG，忽略值 255；类别更多时使用能容纳 ID 的整数格式，如 uint16 TIFF，并在元数据中设置 65535 为忽略值。背景是否属于类别由任务定义，不要直接当作忽略。禁止把 RGB 彩色可视化当类别 ID 标签。建议类别文本使用准确英文名称；任何同义词、类别合并及忽略规则需记录。

这只是准备规范，不是已经实现的自动加载接口。仅图片也能 demo 推理，但不能计算 mIoU。

## 5. 如何转换并检查数据

由师兄依据源格式完成转换。本仓库不新增转换代码；`datasets/prepare_*` 仅用于其指定的公共数据集，不是通用转换器。

检查图片标签数量及一一对应关系、宽高、标签整数值范围、缺失文件、类别顺序，并抽查标签叠加图。转换后须在运行评测的同一进程通过 Detectron2 `DatasetCatalog` 注册记录，每条至少包含 `file_name`、`sem_seg_file_name`；`MetadataCatalog` 设置 `stuff_classes`、`ignore_label`、`evaluator_type="sem_seg"`。参考 `open_vocab_seg/data/datasets/register_ade20k_full.py`。注册名不能冒用已有公共数据集而保留其错误类别。

官方 `train_net.py` 没有命令行导入任意注册文件的功能。师兄需自行让注册代码在该入口运行时导入；注册和评测必须在同一进程，先在另一个 Python 进程注册无效。不能只改 DATASETS.TEST 就当作已接入。

## 6. 小样本测试和全量评测

先运行官方示例，成功标准是得到可打开的分割图片，不只是进程启动：

```bash
CUDA_VISIBLE_DEVICES=0 python demo.py \
  --config-file configs/ovseg_swinB_vitL_demo.yaml \
  --class-names Oculus Ukulele \
  --input resources/demo_samples/sample_03.jpeg \
  --output "$OVSEG_ROOT/outputs/sample_03_prediction.jpeg" \
  --opts MODEL.WEIGHTS "$OVSEG_ROOT/weights/ovseg_swinbase_vitL14_ft_mpt.pth" \
  MODEL.CLIP_ADAPTER.CLIP_MODEL_NAME "$OVSEG_ROOT/weights/ViT-L-14.pt" \
  2>&1 | tee "$OVSEG_ROOT/outputs/demo.log"
```

自有图片可替换 `--input` 为带引号的 glob，并改成任务的候选类别。多图输出需先创建输出目录。demo 只保存可视化，不自动保存原始标签图或计算指标。

全量指标评测需要先完成上一节注册；以下是**条件命令模板，不是开箱即用的自有数据评测**。先注册 custom_small（固定约 10 张）检查，再注册 custom_test 全量；二者类别顺序和忽略规则一致。先使用官方评测默认参数，不根据测试集调参。

```bash
CUDA_VISIBLE_DEVICES=0 python train_net.py --num-gpus 1 --eval-only \
  --config-file configs/ovseg_swinB_vitL_bs32_120k.yaml \
  MODEL.WEIGHTS "$OVSEG_ROOT/weights/ovseg_swinbase_vitL14_ft_mpt.pth" \
  MODEL.CLIP_ADAPTER.CLIP_MODEL_NAME "$OVSEG_ROOT/weights/ViT-L-14.pt" \
  DATASETS.TEST '("custom_small",)' OUTPUT_DIR "$OVSEG_ROOT/outputs/custom_small"
```

全量将 custom_small 替换成 custom_test（包括输出目录）。不要改 `SEM_SEG_HEAD.NUM_CLASSES=171` 来代替注册自有类别；它对应训练模型结构。官方 Pascal Context 和 VOC 使用各自 ensemble weight，详见 GETTING_STARTED.md；自有数据没有已验证的最优权重。使用评测配置而非 demo 配置计算指标，两者 ensemble 设置不应混淆。单卡只改变执行并行度，不主动降低输入尺寸或启用额外 TTA。

## 7. 输入输出、指标、保存位置

模型使用图片和类别文本，将类无关候选掩码与适配 CLIP 的区域分类组合，返回每类像素分数 `sem_seg`（K×H×W）。官方 demo 保存颜色叠加图；官方评测输出到 OUTPUT_DIR/inference，包含 `sem_seg_predictions.json` 和 `sem_seg_evaluation.pth`，日志显示 mIoU、fwIoU、mACC、pACC 及各类 IoU/ACC（百分数）。具体指标沿用仓库 evaluator 实现，缺失类处理需在跨方法比较前核对。没有标注不能报告指标。

本机官方示例 GPU 推理日志为 4.19 秒（非严格测速基准，不含环境安装及模型构建），成功生成可视化；不代表复现了论文数据集分数。

## 8. 数据集、论文和上游出处

固定上游 commit：`36f49d496714998058d115ffb6172d9d84c59065`。

- [论文：Open-Vocabulary Semantic Segmentation with Mask-adapted CLIP，CVPR 2023](https://arxiv.org/abs/2210.04150)
- [官方源码](https://github.com/facebookresearch/ov-seg)，原 README 保存在 README.upstream.md。
- 分割训练 COCO-Stuff-171；CLIP 适配使用挖掘的区域文本对；评测 ADE20K-150/847、Pascal Context-59/459、Pascal VOC-20。数据准备见 datasets/DATASETS.md。
- 上游主体许可证为 CC BY-NC 4.0，部分组件采用独立许可证，见 LICENSE、README.upstream.md 及各子目录。保留全部上游许可，不重新声明整个仓库为 MIT。

## 9. 常见报错

- `np.int` / `np.float` 不存在：确认 NumPy 1.23.5，勿盲目升级。
- CUDA 算子不支持或 undefined symbol：检查 torch/torchvision 配对、CUDA_HOME、nvcc、Detectron2 编译环境；4090 使用 cu118 并针对 8.9 重编译。
- `_C` 导入失败：Detectron2 未正确编译，勿复用另一套 torch 环境的二进制。
- 找不到 ViT-L/14 / home 缓存占满：始终传本地 CLIP 权重路径，不让上游默认下载到 ~/.cache/clip。
- Dataset 未注册：先完成同进程注册，再指定 DATASETS.TEST。图片目录存在不等于已注册。
- OOM：先检查其他进程占用和可用显存；保持官方输入配置，记录图片尺寸。当前不提供多卡模型拆分实现。
- CLIP 提示 PyTorch ≥1.7.1：上游用字符串列表比较版本，1.10 等版本会误报警；本机实际 CUDA 推理已通过，未修改该代码。
- YAML unsafe_load 提示：官方配置含 Python tuple/表达式，只加载可信且固定版本的本仓库配置。
- Google 下载后 torch.load 失败：核验大小及哈希，排除 HTML 页面。
- 下载站证书过期：先核对系统时间和证书链。本机仅对官方 Detectron2 wheel 域名临时使用过 `--trusted-host dl.fbaipublicfiles.com`，未修改全局设置；4090 源码安装不依赖此 wheel 站。

修改与验证边界见 CHANGELOG.md。
