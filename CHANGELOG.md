# 修改记录

基于 facebookresearch/ov-seg commit 36f49d496714998058d115ffb6172d9d84c59065。

- 原 README 原样保存为 README.upstream.md；新的 README 面向 /data/yukun/ov-seg 交接。
- 新增 requirements-4090.txt：未在 4090 验证的 cu118 环境建议；torch 2.0.1 / torchvision 0.15.2 单独安装，Detectron2 v0.6 固定 commit 源码编译。
- 新增本机版本记录、状态文档及忽略规则。
- 没有修改官方 Python、模型配置、分辨率或指标算法；没有新增数据转换、注册或评测程序。数据接入由使用者负责。
- 本机 2080 Ti 官方示例真实推理已验证；4090、自有数据、公共数据集指标未验证。
