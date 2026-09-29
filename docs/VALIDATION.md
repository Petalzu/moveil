# Spark 验证记录（2026-09-29）

## 已执行

- SSH 目标 `spark-95`，主机报告 `spark-3e99`，Linux aarch64。
- 代码 `/home/Developer/moveil-spark`；模型 `/home/Developer/moveil-models/nvidia-gliner-pii`；独立容器 `moveil-gliner`。
- 本次基础镜像本地 ID `sha256:9b2b3cb4d201e48efac830e8b1fd4d5f057be394def6f11cce50f3691d2f8e6a`，无 RepoDigest，**不是可拉取的公开复现引用**。
- torch `2.11.0+cu130`；GLiNER、RapidOCR 实际导入成功；ARM64 wheel 安装成功。
- Windows Python 3.11.9：73/73 单元测试通过，9.772 秒（随后原本地 Python 环境目录消失，不再可用）。
- Spark Python 3.12.13：75/75 单元测试通过，1.529 秒。包含 CUDA 握手拒绝 CPU 回退、自定义模型路径和 UI 设备字段测试。
- 现有 `inspection-vllm` 未停止、未修改；所有安装发生在新容器。

## 真实 GPU 推理与评测

- 权重 1,782,000,995 字节，完整 SHA256 为 `a4dfd0dcbd718acc86dca65fa99f1097d2796c8d7a681e1bc42f40f946c03802`，与固定 revision 预期一致。
- 请求 `cuda`，模型实际参数设备 `cuda:0`，NVIDIA GB10，CUDA 13.0；GLiNER 0.2.29、Transformers 4.51.3。没有 CPU NER 回退。
- 首次实际 smoke：加载 7.421 秒、合成文本预测 0.921 秒。另一次可机器读取的执行证据见 [spark-smoke.json](spark-smoke.json)，耗时以该次记录为准。
- 根样例加 10 张开发图全部真实推理成功，子进程返回 0。11 图处理耗时合计 **7.558 秒**，其中 CPU OCR **5.193 秒**；不含 worker 启动/加载与评测器耗时，不能当端到端总墙钟。
- 开发集：完整遮挡 **52/54（96.30%）**，正确标签完整遮挡 **44/54（81.48%）**，零漏实体图片 **8/10**，最大非敏感墨迹误遮 **11.2575%**。
- 根样例：完整遮挡 **6/6**，非敏感墨迹误遮 **1.5253%**，但正确标签等完整门禁仍未通过。
- **开发集与根样例 gate 均为 false**；email/url 存在漏遮，未修改质量门禁。GPU 适配通过不等于模型质量达标。
- 精简逐图时间、模型文件哈希、审计哈希及指标见 [spark-evidence.json](spark-evidence.json)。原始结果在远端 `runs/spark-gpu-validation-001` 与 `runs/spark-gpu-single-001`。

## UI 实验

- 服务实际仅监听 `127.0.0.1:18198`，经 SSH 隧道在浏览器打开。
- 浏览器点击 10 张样例测试，任务 `live-36538a993e727d85` 完成 **10/10、失败 0、14.5 秒**；页面展示真实 52/54 与 44/54 指标及逐图结果。
- `/api/state` 和运行中的 `/api/batch` 均返回 `actual_device: cuda:0`、`gpu_name: NVIDIA GB10` 及实际版本，不仅检查 HTTP 200。
- 已验证部署文件哈希一致；模型与合成数据保留，权重不进入源码仓库。

## 尚未通过／未验证

基础镜像携带 vLLM/lmcache 等无关包，其依赖与固定 Transformers 4.51.3 冲突；本应用导入成功不代表全镜像 `pip check` 通过。应使用专用运行时，后续记录实际检查结果。

本次不是全新公开镜像安装认证，不是独立 holdout，也未验证中文、复杂扫描或真实业务隐私效果。未声称生产可用；仍需人工复核。

## 发布边界

根 private 目录仅包含公开合成数据的 `truth.json`。保留数据来源、真值、合成 PNG、LICENSE 和用户 NOTICE；删除独立 PCB 53 个文件、旧重复启动脚本，替换无效的拼接 JSON 来源清单。未提交或推送 Git，未处理视频。