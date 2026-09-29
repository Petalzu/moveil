# 第三方数据、模型与依赖声明

## Nemotron-PII 合成数据

- 来源：NVIDIA `nvidia/Nemotron-PII`，2025。
- 作者：Amy Steier、Andre Manoel、Alexa Haushalter、Maarten Van Segbroeck。
- 固定 revision：`b70ffaf5ff39e079776134c5bf4381f00a9fd1ed`。
- 数据集主页：https://huggingface.co/datasets/nvidia/Nemotron-PII
- 许可：Creative Commons Attribution 4.0 International（CC BY 4.0）。许可全文：https://creativecommons.org/licenses/by/4.0/legalcode
- 本目录使用 train offset 50000（根样例）及 50001–50010（开发集）。逐条 UID 和输入哈希见 `SOURCE.json`、开发集 `manifest.json` 与 truth。
- 修改：将合成文本渲染为 PNG，增加独立字符/像素真值及评测统计；未复制原始 record.json。后续运行会生成遮挡图，此类修改不表示原作者认可项目或结果。
- `inputs/sample.png`、根 `private/truth.json`、开发集中的 PNG/truth 及相关数据衍生统计保留上述归属与许可。再次分发时应保留署名、许可链接和修改说明。

## NVIDIA GLiNER-PII

- 模型：https://huggingface.co/nvidia/gliner-PII
- revision：`bd23e8ef4425fd04e34c5204ab49ffaa706eae79`。
- 模型许可：NVIDIA Open Model License；使用前应阅读固定 revision 中的 LICENSE/模型卡及适用条件。
- `pytorch_model.bin` SHA256：`a4dfd0dcbd718acc86dca65fa99f1097d2796c8d7a681e1bc42f40f946c03802`。
- 本仓库不分发权重；`prepare_model.py` 由用户从上游固定版本下载。模型使用的基础组件和依赖可能另有条件，不以本项目源码许可取代。
- 模型训练涉及 Nemotron-PII，随仓开发集不构成独立测试集。

## PaddleOCR 与 RapidAI 转换模型

- 上游模型：PaddlePaddle/PaddleOCR，https://github.com/PaddlePaddle/PaddleOCR 。原始模型权利归 Baidu 及相应 PaddleOCR 权利人所有。
- 推理软件及资产发布：RapidAI/RapidOCR，https://github.com/RapidAI/RapidOCR 。软件版权归 RapidOCR Authors；RapidOCR 上游声明其 PaddleOCR 派生权重及转换资产沿用 Apache-2.0 条款。
- PyTorch 转换项目致谢：PaddleOCR2Pytorch，https://github.com/frotms/PaddleOCR2Pytorch ，采用 Apache-2.0；RapidOCR 在上游致谢中列为 Torch 转换模型来源。
- 本项目固定 RapidOCR **3.9.2**，模型发布版本 **v3.9.2**，来源为 ModelScope 的 `RapidAI/RapidOCR`。使用 PP-OCRv5 Server `ch_PP-OCRv5_rec_server.pth`、PP-OCRv6 Medium `PP-OCRv6_rec_medium.pth` 及对应字符字典。
- 权重与字典的文件名、SHA256 固定于 `ocr_backend.py`；`prepare_model.py --ocr` 下载并核验，推理加载前再次校验。本项目使用上游转换资产进行推理，保留模型权重与字典原内容。
- 软件、模型及转换资产按各自适用的 Apache License 2.0 条款使用，许可全文见 https://www.apache.org/licenses/LICENSE-2.0 。分发模型资产时应同时保留上游许可、版权归属及适用 NOTICE，并注明实际修改。

## 软件依赖与历史研究代码

Pillow、NumPy、Requests、RapidOCR、ONNX Runtime、GLiNER、PyTorch、Transformers、Hugging Face Hub 及其传递依赖归各自权利人所有，按各自发布许可使用；本项目通过 requirements 安装，不重打包其源码或模型。RapidOCR 的随包模型也应按上游条款使用。

共享研究模块仍保留 Gretel 加载逻辑；相应权重未分发，也不属于默认复现路线。若启用该路线，应额外核查上游许可。PCB/VisA/Qwen 示例已从此应用交付目录移除，不分发相应数据或权重。使用外部 CUDA/PyTorch 容器时，必须保留镜像自带的第三方许可。

项目自有源码已由用户确认采用 Apache-2.0，见 `LICENSE`、`NOTICE`。本许可不替代上述第三方条款，也不是 NVIDIA 对整个项目的授权或认可。

演示截图、遮挡图和视频包含上述合成数据衍生内容，须保留 CC BY 4.0 署名、许可链接及修改说明。媒体与制作脚本独立交付，成片说明及本声明副本随视频材料保留。媒体制作工具额外使用 Playwright、FFmpeg、Pillow、Windows 字体及历史可选语音引擎；它们按各自许可使用，不随仓重打包其软件、字体或模型，也不加入应用运行依赖。