# 复现与交接状态

上游：https://github.com/facebookresearch/ov-seg

固定上游 commit：`36f49d496714998058d115ffb6172d9d84c59065`。

## 本机已验证

RTX 2080 Ti，驱动 545.29.06；Python 3.8.20、torch 1.10.1+cu113、torchvision 0.11.2+cu113、Detectron2 0.6、NumPy 1.23.5、Pillow 9.5.0、timm 0.6.13。官方 Swin-B + ViT-L/14 完整权重在官方 sample_03.jpeg 上完成真实推理，日志记录 4.19 秒，输出已目视检查。源码、分辨率和评测参数未修改。

OVSeg 权重 MD5：526080bf24e7182a8b5df4b92c5a14ec。

原始 CLIP ViT-L/14 SHA-256：b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836。

## 未验证

RTX 4090 环境、自有数据和论文指标均未验证。师兄根目录使用 /data/yukun/ov-seg，不使用 /data/yipu。

## 数据接口限制

官方 train_net.py 通过 DatasetCatalog、MetadataCatalog 加载预注册数据集，没有通用的任意自有数据命令行入口。现有 datasets/prepare_* 工具面向指定公共数据集，不能直接当作任意彩色标签或 JSON 的转换器。

自有语义分割数据需明确图片与单通道类别 ID 标签的对应关系、类别名称和 ID、忽略值，以及测试集划分。没有真值标签时只能推理，不能计算 mIoU。背景是否为评测类别必须由数据定义，不能擅自忽略。

## 修改记录

增加交接 README、依赖建议、本机版本记录、修改记录和忽略规则；模型源码保持上游原样。按用户要求不新增自有数据注册、转换、检查或评测代码；这些工作由师兄完成。详见 CHANGELOG.md。
