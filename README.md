# 墨隐 MoVeil：DGX Spark 图像隐私识别与辅助脱敏

对文档截图进行 **CPU OCR → NVIDIA GLiNER-PII GPU 实体识别 → 字符/像素映射 → 永久遮挡 → 人工复核**。这是辅助脱敏原型，不保证自动清除全部隐私。

## Spark 部署

要求 Linux aarch64、NVIDIA Container Toolkit 和支持 GB10 的 CUDA PyTorch。使用独立容器，不修改已有推理服务。保留基础镜像 CUDA torch；`requirements-spark.txt` 不安装 CPU torch。

1. 代码挂载到 `/app`，工作目录 `/app`；完整模型目录挂载到 `/models/nvidia-gliner-pii`。
2. 设置 `MOVEIL_DEVICE=cuda`、`MOVEIL_NER_PYTHON=/usr/bin/python3`、`MOVEIL_MODEL_DIR=/models/nvidia-gliner-pii`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`。解释器路径以实际 CUDA 环境为准。
3. `bash start.sh setup` 安装依赖；可通过 `PIP_INDEX_URL` 指定可信 HTTPS 镜像，不关闭 TLS 校验。准备模型时挂载可写，执行 `bash start.sh prepare`；推理时改为只读挂载。
4. `bash start.sh check` 校验权重、CUDA 张量、模型实际设备及合成文本真实预测，请求 CUDA 时拒绝 CPU 回退。
5. `bash start.sh serve --port 18198` 先对根样例推理，成功后启动回环 UI；`--run runs/已验证目录` 可复用结果。

已有准备好的 ARM64 CUDA 运行时镜像时，设置 `MOVEIL_IMAGE`（建议固定 registry digest）、`MOVEIL_HOST_MODEL_DIR`（宿主完整模型目录），执行 **`bash spark.sh --port 18198`**。脚本不猜测镜像、不在线安装依赖。带 vLLM 等无关包的基础镜像可能存在依赖冲突，不能把导入成功当作全镜像依赖检查通过。

本机建立 `ssh -N -L 18198:127.0.0.1:18198 spark-95` 后访问 `http://127.0.0.1:18198`。应用只绑定回环，不需要公网开放端口。

`start.sh` 读取根 `.env`；配置示例见 `.env.example`。避免旧配置覆盖容器路径。直接运行 Python 需自行导出环境变量；Python 默认允许 CPU 研究，Spark shell 入口默认 CUDA。

## 固定模型与证据

- `nvidia/gliner-PII` revision：`bd23e8ef4425fd04e34c5204ab49ffaa706eae79`。
- 权重 SHA256：`a4dfd0dcbd718acc86dca65fa99f1097d2796c8d7a681e1bc42f40f946c03802`。
- 下载器和 worker 共用 `MOVEIL_MODEL_DIR`，默认 `.venv-ner/nvidia-gliner-pii`；必须包含完整 tokenizer/config。
- worker 离线加载；审计记录实际设备、版本、模型/源码/输入输出哈希。72 词窗口、18 词重叠、阈值 0.3、24 标签分组。

本次验收见 [docs/VALIDATION.md](docs/VALIDATION.md)。历史 `cohorts/nvidia-gliner-pii-dev-015/summary.json` 是 **Windows CPU** 结果（52/54 覆盖且门禁失败），不是 Spark GPU 证据。

本次 Spark 实测：GB10 `cuda:0` 完成 11 图，75 项单元测试通过；开发集完整遮挡 52/54、正确标签完整遮挡 44/54，**质量门禁未通过**。11 图处理合计 7.558 秒（不含加载），UI 另行完成 10 图真实批处理。详见本次证据，不以历史 CPU 指标代替。

真实 11 图复现：`python3 nvidia_ner_benchmark.py runs/spark-gpu-validation-001 --from-cohort cohorts/nvidia-gliner-pii-dev-015 --single-out runs/spark-gpu-single-001`。使用全新输出目录；保存逐图审计、耗时、像素指标与门禁结果。OCR 使用 CPU，NER 使用 GPU。

单元测试：`python3 -m unittest test_agent test_pipeline test_pii_rules test_ner test_nvidia_ner test_review test_release test_prepare_dataset test_launch test_hybrid_geometry`。mock 不替代真实推理。

## 交互与安全

支持 1–10 张合成样例、单图上传、并排复核、人工补框、版本化保存及 PNG 导出。上传限制为 10 MiB、PNG/JPEG/WebP 单帧、边长 8192、总像素 16,000,000，后台串行处理。

校验 Host、Origin、Sec-Fetch-Site、写入令牌与路径；页面禁缓存，导出剥离元数据。结果始终需要逐图人工复核。像素遮挡验证不证明没有漏检，质量门禁失败不能标为生产可用。`runs/`、上传、权重和 `.env` 默认忽略；审计仍需按敏感材料管理。

## 数据、代码与许可

随仓 11 张 PNG 来自 NVIDIA Nemotron-PII **公开合成数据**，真值仅用于评测、不传入模型。开发集与模型训练来源重叠，不是独立测试集；中文、复杂版面及真实业务数据未验证。`SOURCE.json` 与开发集 manifest 保留来源。`prepare_dataset.py --output data-prepared` 可准备数据，但像素重建依赖指定 Windows Consolas 字体及 Pillow；Spark 直接使用随仓 PNG。

主流程为 `agent.py`、`hybrid_geometry.py`、`ner_pipeline.py`、`nvidia_ner_*`；交互为 `review.py`、`live_batch.py`、`web/`；评测为 `evaluate.py`、`benchmark.py`。保留共享 NER 模块，移除独立 PCB 历史目录与重复启动脚本。exact/Gretel 不是默认 GPU 路线，启用远端文本服务前需评估隐私风险。

源码采用 [Apache-2.0](LICENSE)，保留用户 [NOTICE](NOTICE)。合成数据遵循 CC BY 4.0；模型遵循 NVIDIA Open Model License，权重不随仓发布。归属见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。