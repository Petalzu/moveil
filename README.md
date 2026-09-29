# 墨隐 MoVeil：DGX Spark 图像隐私识别与辅助脱敏

## 1. 项目目标

MoVeil 面向文档截图中的个人敏感信息，构建从文字识别、语义判断到像素遮挡、人工复核与结果导出的本地工作流。项目采用 CPU RapidOCR 与 NVIDIA GPU GLiNER-PII 协同处理，将实体识别结果映射到图像坐标，生成可复核、可追加遮挡、可追溯的处理版本。

项目定位为图像隐私辅助脱敏工具，最终发布由业务人员结合使用场景和授权范围确认。

## 2. 应用场景

- 工单与问题反馈：处理截图中的姓名、联系方式、账号和地址，再交付协作人员。
- 文档展示与培训：对静态图片中的敏感字段进行遮挡，保留版面与业务上下文。
- 数据协作与资料分享：在本地完成识别和复核，按用途导出处理后的图片。
- 隐私处理流程演示：使用随仓公开合成样例展示 GPU 识别、像素覆盖和人工修订流程。

## 3. 功能与交互

1. 选择 1–10 张合成样例，触发实时 GPU 推理，逐张查看处理进度与结果。
2. 上传 PNG、JPEG、WebP 单帧图片；容量上限为 **10 MiB**，边长上限为 **8192 像素**，总像素上限为 **16,000,000**。
3. 单屏工作台集中展示任务、指标与编辑工具，原图和处理图上下对照；支持分页、检测框与标签切换，以及适应窗口、铺满宽度、原始尺寸三种显示模式。
4. 拖动矩形补充遮挡，实时预览并撤销草稿操作；切换页面时保留各图草稿。
5. 保存补遮挡新版本，保留已有黑色遮挡及父版本审计关系，支持连续修订。
6. 下载清除元数据的 **RGB PNG**，将复核后的像素遮挡结果用于后续协作。

### 界面展示

![MoVeil 单屏编辑工作台：左侧任务与指标，右侧原图和处理图上下对照，顶部提供补框保存与下载操作](docs/images/workspace-overview.png)

图 1：DGX Spark 实际运行界面，采用随仓公开合成样例 `inputs/sample.png`，展示识别与人工补框保存后的状态，截图分辨率为 **2376 × 1485**。左侧集中展示任务进度和识别数据，右侧上下排列原图与处理图，编辑工具持续可见；数据面板支持收起，实体明细按需展开，放大图片时在画布视口内滚动。带真值样例展示像素覆盖指标，上传模式展示识别结果与复核状态。

## 4. 技术架构

处理链路：**图片校验与规范化 → CPU OCR → GPU PII 识别 → 字符与像素对齐 → 黑色像素遮挡 → 人工复核 → 版本化导出**。

- 输入层：校验格式、尺寸和帧数，处理 EXIF 方向，转换为 RGB 图像。
- 识别层：RapidOCR/ONNX Runtime 在 CPU 上提取文本与位置；GLiNER-PII 在 GPU 上完成语义实体识别，批内复用模型 worker。
- 通信层：通过本地 JSONL 子进程协议传递请求与响应，校验实际设备、请求序号与配置指纹，并记录异常状态。
- 遮挡层：校验实体跨度、标签和置信度，结合几何对齐将字符范围投射到图像，绘制实心黑色像素。
- 交互层：回环 HTTP 服务配合原生 HTML/CSS/JavaScript 画布，实现串行任务、结果浏览和人工补框。
- 审计层：记录运行配置、设备、软件版本及模型、源码、输入输出的 SHA256；人工修订保存父审计哈希和像素核验结果。

## 5. 模型与运行参数

- 模型：`nvidia/gliner-PII`。
- 固定 revision：`bd23e8ef4425fd04e34c5204ab49ffaa706eae79`。
- 权重 SHA256：`a4dfd0dcbd718acc86dca65fa99f1097d2796c8d7a681e1bc42f40f946c03802`。
- 文本窗口：**72 词**；相邻窗口重叠 **18 词**；子词预算 **352**。
- 标签分组：每组最多 **24 个标签**；置信度阈值 **0.3**；推理 batch size **1**。
- 加载方式：由 `MOVEIL_MODEL_DIR` 指定完整模型目录，包含权重、tokenizer、配置和 SentencePiece 资源，推理采用本地离线加载。

## 6. 部署与启动

**部署配置**：复用具备 CUDA PyTorch 和应用依赖的 `moveil-gliner` 容器；容器目标目录为 `/opt/moveil-core`，宿主机备份目标目录为 `/home/Developer/moveil-core`，服务端口为 **18200**。

在具备运行依赖的环境中，于项目目录设置变量后启动：

```bash
export MOVEIL_DEVICE=cuda
export MOVEIL_NER_PYTHON=/usr/bin/python3
export MOVEIL_MODEL_DIR=/models/nvidia-gliner-pii
export MOVEIL_LOAD_DOTENV=0
bash start.sh serve --port 18200
```

代码部署到目标目录后，在 Spark 宿主机中使用已有容器启动：

```bash
docker exec -it -w /opt/moveil-core \
  -e MOVEIL_DEVICE=cuda \
  -e MOVEIL_NER_PYTHON=/usr/bin/python3 \
  -e MOVEIL_MODEL_DIR=/models/nvidia-gliner-pii \
  -e MOVEIL_LOAD_DOTENV=0 \
  moveil-gliner bash start.sh serve --port 18200
```

该入口先核验运行环境并处理根样例，再启动复核页面。`MOVEIL_LOAD_DOTENV=0` 采用显式环境变量；启动脚本清空 `PYTHONPATH`，设置 `PYTHONNOUSERSITE=1` 和 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。

本机通过 `ssh -N -L 18200:127.0.0.1:18200 spark-95` 建立 SSH 转发，再访问 **http://127.0.0.1:18200**。

**首次环境准备**：在 Linux aarch64 上配置 GPU 驱动与支持 GB10 的 CUDA PyTorch；容器方式同时配置 NVIDIA Container Toolkit。设置上述环境变量后，依次运行 `bash start.sh setup` 安装应用依赖、`bash start.sh prepare` 下载固定版本模型。日常运行使用 `serve` 入口。

**镜像启动方式**：具备完整应用依赖的 ARM64 CUDA PyTorch 镜像可通过 `MOVEIL_IMAGE` 指定，宿主机模型目录通过 `MOVEIL_HOST_MODEL_DIR` 指定，再执行 `bash spark.sh serve --port 18200`。

## 7. 开发集结果与指标

核心版本在 DGX Spark 上完成根样例与 10 图开发集推理，**11 张图片全部成功**。固定公开合成开发集包含 **54 个标注实体，失败记录为 0**：

- 实体完整遮挡：**52/54 = 96.30%**。
- 正确标签完整遮挡：**44/54 = 81.48%**。
- 零漏实体图片：**8/10**。
- 最大单图非敏感墨迹误遮比例：**11.2575%**。

对应实测环境为 **NVIDIA GB10 / Linux aarch64**，驱动 **580.82.09**，Python **3.12.13**，PyTorch **2.11.0+cu130**，CUDA **13.0**，GLiNER **0.2.29**，Transformers **4.51.3**；模型执行设备为 `cuda:0`。

页面实测覆盖 10 图批处理、分页、拖框、草稿保留、撤销、版本保存、下载与上传识别。10 图任务耗时 **14.1 秒**，单图上传识别耗时 **8.4 秒**；计时口径为本次任务从启动到完成的墙钟时间，包含 worker 加载与图像处理。下载 PNG 与页面处理图的 SHA256 一致，补框区域通过纯黑像素核验。

核心交付包含 **15 个 Python 运行模块**，本地源码、Spark 宿主机备份和运行容器逐文件 SHA256 一致。批处理记录位于 `runs/live-77aa07f5b6f3170f/`，根样例记录位于 `runs/bootstrap-c99b002135def490/`，运行产物保存在 Spark 部署目录。

指标基于输出图像的实际像素计算。原图任一 RGB 通道小于 250 的像素计为墨迹，真值框内的墨迹计为敏感墨迹。实体完整遮挡要求该实体全部敏感墨迹落入遮挡区域且输出像素为纯黑；正确标签完整遮挡进一步要求相同标签的遮挡框覆盖这些像素。两项比例均以标注实体总数为分母。

非敏感墨迹误遮比例为真值敏感区域之外、落入遮挡区域的墨迹像素数占全部非敏感墨迹像素数的比例。零漏实体图片比例反映单图全部标注实体获得完整遮挡的情况。

`quality_gate.json` v2 要求图片数 **≥10**、失败记录 **为 0**，且实体完整遮挡率与正确标签完整遮挡率均**严格大于 80%**。统计对象为模型自动输出，人工补框作为独立复核版本留存；上述开发集记录达到该门槛。

开发集与模型训练数据同源，结果用于固定合成开发集上的工程质量分析。后续将使用独立、经授权的业务数据开展评估，覆盖中文、复杂版面及真实业务字段，并重点改进 email/url 识别与标签映射。

## 8. 核心文件

- `agent.py`、`hybrid_geometry.py`：OCR、输入与结果校验、几何对齐及像素遮挡。
- `ner_config.py`、`ner_worker.py`、`ner_pipeline.py`：标签映射、文本分窗、识别流水线与 worker JSONL 通信。
- `nvidia_ner_config.py`、`nvidia_ner_worker.py`、`nvidia_ner_pipeline.py`：NVIDIA 模型参数、GPU 推理与流水线封装。
- `evaluate.py`、`quality_gate.json`：像素覆盖计算、逐图测量、批量汇总与质量门槛。
- `live_batch.py`、`review.py`、`web/`：样例批处理、图片上传、HTTP 服务与复核界面。
- `run_demo.py`、`launch_check.py`、`prepare_model.py`、`tools/check_gpu.py`：应用入口、启动核验、模型准备与 GPU 环境检查。
- `start.sh`、`spark.sh`、`requirements.txt`、`requirements-spark.txt`：运行脚本、容器入口和依赖清单。
- `inputs/`、`private/truth.json`、`cohorts/nvidia-gliner-pii-dev-015/`：根样例、像素真值与 10 图合成样例集。
- `docs/images/workspace-overview.png`：单屏编辑工作台实机截图。

## 9. 数据治理

OCR、实体识别和像素遮挡在运行机器内完成，浏览器通过本机或 SSH 隧道访问回环服务。服务校验 Host、Origin、Sec-Fetch-Site 与写入令牌，按资源路径约束访问；导出图片清除元数据。

模型处理图片及 OCR 文本，真值数据进入独立的指标计算流程。人工复核负责确认敏感字段、补充遮挡并选择发布版本；每项产物通过 SHA256 与来源关联，修订版本通过父审计哈希连接。

部署管理应为原图、上传文件、OCR 文本、审计记录和输出版本设置访问权限，按授权用途制定保留期限、备份策略与到期清理流程，并将运行目录纳入数据生命周期管理。

## 10. 数据来源与许可

随仓图像来自 NVIDIA `nvidia/Nemotron-PII` 公开合成数据，固定 revision 为 `b70ffaf5ff39e079776134c5bf4381f00a9fd1ed`；根样例取自 train offset 50000，10 图开发集取自 50001–50010。项目将合成文本渲染为 PNG，并增加字符与像素真值；来源、记录标识及哈希见 `SOURCE.json`、`SOURCE-MANIFEST.json` 和 `cohorts/nvidia-gliner-pii-dev-015/manifest.json`。

源码采用 **Apache-2.0**，许可及归属声明见 `LICENSE`、`NOTICE`；数据采用 **CC BY 4.0**；模型采用 **NVIDIA Open Model License**，权重通过固定上游版本单独获取。数据作者为 Amy Steier、Andre Manoel、Alexa Haushalter、Maarten Van Segbroeck，来源机构为 NVIDIA。分发衍生图像时应保留署名、许可及修改说明；模型、软件依赖和容器组件分别遵循各自许可，详细归属见 `THIRD_PARTY_NOTICES.md`。