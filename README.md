# Paper Agent v1.0.0

当前课程提交候选版采用按排名分批等待、失败后顺序补位的全文获取方式（batch）。设置页“下载与全文处理 → 时间预算方案”提供默认、慢速网络和自定义，分别控制单个 PDF 地址、单篇下载与整个全文阶段；默认为 45/180/180 秒，慢网为 90/240/600 秒。三项均可自行修改，保存后用于新任务。滚动替补、短超时两阶段和断点续传未纳入当前运行版；现有实验不能证明它们在真实网络下稳定更快。

本次交付范围与合并步骤见[最终提交准备](docs/最终提交准备_20261113.md)，精简测试与历史对照结果见[验证摘要](docs/最终提交验证摘要_20261113.json)。源码包含实现、测试及方法说明；本机 `data/` 原始实验、用户历史、模型缓存和相邻源码快照不会随 Git 自动提交，复现实验须另行准备对应输入。

设置页现在按论文来源、候选排序、全文转换、单篇分析提供独立开关；同一环节的分析方案互斥，跨环节可组合，点击“保存研究方法”后用于新任务。原预设继续兼容历史与 CLI。操作、原九项含义和修改前回档快照见[独立开关说明](docs/研究方法独立开关_20261112.md)。

“全文证据分析”（`grounded`）已接入：完整保留已提取原文，一次调用先提取引句再作答，核对引用位置并保留真实材料范围。工程验证通过，真实质量验证因生成接口故障尚未完成，暂不能宣称准确性或速度提升。固定开发/留出设计、开源来源、评分纠错与复现见[质量验证记录](docs/全文证据分析与回答质量验证_20261107.md)。

新增“级联证据检索”实验档位（`evidence_fast`）：BM25 初筛后对前 40 个子块联合重排，复用父段恢复与原文引用。98 篇新留出中证据覆盖率 89.11%，独立新文档检索计时较原混合方法减少 52.8%。旧 12 题回答评分遗漏了可拒答注释，已按官方规则纠正：级联 F1 57.81%、原混合 60.72%、整篇 54.02%；旧“级联低于整篇”的结论撤回，整篇速度仍更快。另修复 arXiv 不同版本共用 PDF、关键词及独立 Markdown 缓存的问题。完整结果、纠错与复现见[级联实测](docs/级联证据检索与版本缓存实测_20261107.md)。

篇内证据检索已提供独立对照档位，采用混合召回、联合重排与父段恢复，并核对原文引句及字符位置。QASPER 100 篇留出评估和 12 题真实模型配对显示输入用量减少，端到端速度暂未提升。机制、数据、限制与运行方法见[篇内证据实测](docs/篇内证据检索与引用回查实测_20261107.md)。

基线对照：设置页“研究方法开关”可逐项切换基线与新增方法，新任务记录配置与输入指纹，各次 Markdown/报告独立保存。CLI 支持 `--profile`，`scripts/compare_methods.py` 可将保存的真实测量整理为独立展示包，`--run` 重做免费本地模块对照。用法和可声明的结果见[基线与新方法对照指南](docs/基线与新方法对照指南_20261107.md)。

整篇论文解析现默认使用已安装的 CPU Layout 引擎，恢复阅读顺序、章节和表格后仍只做一次单篇分析；异常时回退旧解析器。三篇真实 PDF 的两轮转换对照分别缩短约 57%、15%、22%，不等同于整套研究或 LLM 推理提速。开源项目调研、实表核对与复现见[开源机制复用记录](docs/开源机制复用与整篇解析改进_20261106.md)。`PAPER_AGENT_PDF_PARSER=legacy` 可使用原解析器。

免费来源扩展：真实研究默认额外查询 Crossref 期刊与会议题录，无需注册或密钥。其失败不阻断原有 arXiv/OpenAlex 结果，来源计数、独有候选与失败状态见“执行记录 → 数据来源”；`PAPER_AGENT_CROSSREF_ENABLED=false` 可关闭。Crossref 全文链接不自动视为开放全文。真实覆盖记录与证据边界见 [免费来源接入记录](docs/免费来源接入记录_20261106.md)。

手写参考价值排序与长文分段并发均为默认关闭的实验选项，前者未证明值得采用的收益，后者三篇真实模型首轮均更慢、总 tokens 增加约 75%；详见[原实验记录](docs/排序与篇内并发记录_20261106.md)。随后复用开源联合重排机制，两组各 20 题的候选 nDCG@5 分别从 0.719→0.807、0.749→0.916，原前二十篇集合保持不变；`PAPER_AGENT_CROSS_ENCODER_ENABLED=true` 可开启本地可选策略，默认关闭，不与手写价值规则叠加。该证据还不代表最终 LLM 精选或报告质量，见[开源机制复用记录](docs/开源机制复用与整篇解析改进_20261106.md)。

Paper Agent 是本机运行的论文研究桌面应用。当前界面使用 **Electron + React + TypeScript + Vite**；Python 后端保留 MainAgent、MCP Client/Server、arXiv 与 OpenAlex 来源、PDF 全文处理、Selector 和 Research Analyst。renderer 只经 preload API 与 Electron 主进程通信；真实研究仍走 MainAgent → MCP Client → MCP Server，不绕过既有工具链。

版本变更摘要见 [CHANGELOG](CHANGELOG.md)，当前交付与已知限制见 [v1.0.0 交付说明](docs/v1.0.0-交付说明.md)，本次顺序/并行真实实验见[实验记录](docs/并行优化真实实验_20261102.md)。完整历史与证据边界见[开发日志](docs/开发日志.md)。角色与文件所有权见[分工计划](docs/分工计划.md)，接口见[架构与接口](docs/架构与接口.md)，安装/验收细节见[运行与验收](docs/运行与验收.md)。公共模型与函数规则见[开发契约](docs/开发契约.md)和[schema](docs/contracts.schema.json)。

工作区支持研究与基于已有资料的讨论、会话历史、论文与报告阅读、真实工具事件、模型设置和 artifact 打开/导出。默认展示关键发现、关键论文和研究下一步；来源、原始 Top-k、全文质量、候补及校验记录在执行记录中按需查看。

当前桌面研究对全部去重候选完成语义融合排序，再把前 20 篇论文的完整标题和摘要交给任务 Agent 选出 Top-k；不足 20 篇时全部传入。其余候选仍按原排名保留，供 PDF 获取失败或转换没有可读正文时向后替补。默认最多尝试 `2 × Top-k` 篇后备论文，整个全文获取与转换阶段限时 180 秒；已有部分正文、原始选择和失败记录均保留。可通过 `PAPER_AGENT_MAX_FULLTEXT_ATTEMPTS`（0 表示自动上限）和 `PAPER_AGENT_FULLTEXT_TIMEOUT` 调整。默认单次筛选输入上限为 256000 个保守 UTF-8 字节估算单位；超限时在这最多 20 篇中分批完整评估并汇总终选。语义输入可加入已有 PDF 中明确标注的作者关键词，不使用自动主题词冒充作者关键词。当前规则见[开发契约](docs/开发契约.md)。

默认分析链路为：每篇原文 Markdown → 一次 Agent 分析 → 单篇报告；全部单篇任务结束 → 将完整单篇报告一起提交给 Agent → 一次总体报告。并行开关控制多篇单篇报告是否同时生成。显式配置 `PAPER_AGENT_PAPER_ANALYSIS_STRATEGY=sections` 时，至少 40000 字符的已提取原文才尝试三段分析后综合，共用实际模型请求并发上限；本轮首测未达到速度、用量与质量采用门槛。单篇和总体输入默认分别允许 256000 个保守 UTF-8 字节单位，显式预算配置优先，超限时保留失败原因，不静默截断。v1.0.0 已发布实验反映原版本流程，不作为本次改动后的耗时测量。

v1.0.0 在输入区提供“并行优化”开关：关闭顺序执行，开启后同时检索来源并处理多篇论文，并发数可在设置中调整。性能页展示实际执行方式、阶段耗时和两种 Top-k 分数。研究笔记、原文 Markdown 与 PDF 原文提供同级阅读入口。真实研究仍走多源检索、Sentence-BERT 混合排序和 LLM 精选；部分分析继续保留真实状态。

后续开发统一使用长期分支 `dev`，通过 PR 提交到 `main`，不创建版本号命名的开发分支。应用版本号独立维护。

## 快速开始

开发环境需要 Python 3.11+、Node.js 22+ 和 npm。在 Windows PowerShell 的项目根目录安装依赖：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1
```

该脚本准备 `.venv-v2`、Python 开发/打包依赖和 root npm workspaces。手工安装：

```powershell
python -m venv .venv-v2
.\.venv-v2\Scripts\python.exe -X utf8 -m pip install -e ".[dev,packaging]"
npm ci
```

语义排序需要另行安装可选依赖并提前缓存模型：`.venv-v2\Scripts\python.exe -m pip install -e ".[semantic]"`。当前机器已安装 CPU 语义运行环境并实际验证；环境缺失时会如实回退关键词排序，不应把回退结果称作语义对照。具体版本和离线准备方式见[实验记录](docs/排序与篇内并发记录_20261106.md)。

启动源码开发模式（Vite HMR + Electron）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\dev-desktop.ps1
```

构建 renderer/Electron 并从源码打开应用：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start-desktop.ps1
```

该脚本执行 `npm run build` 后启动应用。Windows 发布包由 `scripts/build-desktop.ps1` 构建：它先打包 Python backend，再生成 electron-builder 配置的 Windows x64 NSIS 与 portable 安装包。macOS 构建和运行尚未在本轮验证。

## 研究模式

默认离线 Demo 使用明确标记的合成资料和响应，可在不调用真实模型的情况下核对 MCP 工具链和业务组合：

```powershell
.\.venv-v2\Scripts\python.exe -X utf8 -m paper_agent.cli --mode demo --k 5
```

Live 模式会请求真实论文来源和模型，可能产生 API 用量。配置 DeepSeek key 后再使用；OpenAlex key 可选。Demo 结果不能证明论文相关性、模型质量或真实性能提升。发生来源、全文、模型或校验错误时，系统保留 partial/failure 状态与已有产物。

## 本地保存 DeepSeek 与 OpenAlex Key

1. 在 [DeepSeek 开放平台](https://platform.deepseek.com/)创建/查看 DeepSeek API Key；需要 OpenAlex 个性化鉴权时，在 [OpenAlex API 设置](https://openalex.org/settings/api)查看自己的 key。
2. 打开应用的“设置”页，在 DeepSeek API Key 中粘贴自己的 key；如要使用 OpenAlex 个性化鉴权，再填入自己的 OpenAlex key。
3. 点击“保存设置”。默认模型为 `deepseek-flash`，API 根地址为 `https://api.deepseek.com`，thinking 默认关闭。保存只更新本机文件并显示是否已配置，不会调用模型验证，也不会回显 key。

Windows 默认 data root 与配置文件为：

```text
data_root:   %LOCALAPPDATA%\Assignment\PaperAgent
config file: %LOCALAPPDATA%\Assignment\PaperAgent\config.json
```

文档和命令中的 key 值只用占位说明，不粘贴真实凭据。配置文件包含用户保存的密钥明文，请勿分享、提交或上传。密钥框留空会保留原值；使用设置页的 clear 操作才会清除。

优先级按当前实现执行：桌面 `config.json` 中已保存的字段优先于进程环境；缺少配置字段时回退环境。进程环境优先于 `.env`；非空 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL` 优先于对应 `DEEPSEEK_*`。OpenAlex 使用 `OPENALEX_API_KEY`。CLI 使用进程环境和 `.env`，不读取桌面 `config.json`。

需要另一个资料目录时，设置 `PAPER_AGENT_DESKTOP_DATA_DIR`。应用、服务和服务控制进程应使用相同值。当前打包配置不把 `.env`、`config.json` 或 user data root 加入发布资源，也不要求将密钥交给构建过程；key 在用户首次运行后写入本机资料目录。

## 任务、重连与停止

Python 服务只监听 `127.0.0.1`，使用 Bearer 鉴权。renderer 不持有服务 token。窗口关闭或 Vite HMR 只断开界面，不会取消由服务持有的研究任务；重新打开会从同一 data root 读取会话、任务和事件。要终止任务，使用应用中的显式取消操作。

服务停止命令：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\manage-service.ps1 -Action status
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\manage-service.ps1 -Action stop
```

有 queued/running 任务时，服务会拒绝停止。先等待任务结束，或从应用中显式取消任务；不要强制结束服务进程。原有 session/job 若缺少 `conversation_id`，仍会出现在历史中并标记 `read_only: true`；旧记录可读，但不能继续追加消息。详见[本机服务与历史接口](docs/架构与接口.md)。

## 检查与验收

本地检查命令：

```powershell
.\.venv-v2\Scripts\python.exe -X utf8 -m pytest -q
.\.venv-v2\Scripts\python.exe -m ruff check src tests scripts
npm run typecheck
npm run test:frontend
npm run test:electron
npm run build
npm run test:e2e
```

较大数据量档位（40/40/30/8）使用独立的、被 Git 忽略的目录；demo 档位不调用模型，live 档位必须显式确认：

```powershell
.\.venv-v2\Scripts\python.exe -X utf8 scripts\stress_run.py --dry-run --mode demo
.\.venv-v2\Scripts\python.exe -X utf8 scripts\stress_run.py --mode live --yes `
  --k 8 --source-limit 40 --candidate-limit 30 --merged-limit 30 `
  --max-llm-calls 160 --max-total-tokens 1200000 --max-chunks 10 `
  --report-budget 128000 --download-timeout 90
```

结果写入 `data/validation/v021/stress/<时间>/stress-summary.json`，包含每阶段耗时、真实计数、错误和 tool call 记录。demo 结果只能证明链路组合，不能证明真实论文质量或性能。

CI 执行 Python 测试/Ruff 与 npm typecheck/build/Electron Node 测试。代码通过、类型检查和自动测试不能代替已安装桌面应用的端到端交互检查，也不能替代 live 论文质量验收。

### 历史验证范围（v0.3.0）

版本、构建检查与真实研究限制见 [v0.3.0 交付说明](docs/v0.3.0-交付说明.md)。本次没有新增研究实验，也没有重新运行测试套件；已有 PDF 阅读核对与真实研究结果分别记录。全文获取、重复结论与引文匹配仍有未解决的问题，不能把界面改善当作研究质量已经完整通过。本版没有重新生成安装包。

### 历史验收记录（v0.2.2）

| 范围 | 实际记录 |
|---|---|
| Python 单测 | 307 passed、3 skipped（外网检查默认跳过） |
| Ruff | `ruff check src tests scripts` 通过 |
| TypeScript / 构建 | `npm run typecheck`、renderer/main/preload 生产构建通过 |
| Electron / 前端单测 | Electron 8/8；前端 layout/labels/activity/papers 8/8 |
| 桌面 E2E（源码） | 16 项通过：菜单移除、Sidebar、Dock 溢出/缩放、活动卡片、论文跨会话恢复、输入框自增高、取消、窄窗口 |
| 桌面 E2E（打包） | 16/16，解包版实际启动并通过完整流程 |
| Windows 发布包 | v0.2.2 解包版已生成并通过 E2E；NSIS/portable 安装包本轮未重新生成 |
| macOS 构建/运行 | 未验证 |
| live 研究 | v0.2.2 小规模真实运行用于中文输出/PDF 预算核验；结果按真实失败保留为 partial，不宣称完整科研成功 |
| live 性能结论 | 未做同输入 ≥5 轮配对基准；不提供加速比 |

历史真实 API 检查和 Top-1 研究的 partial 结果见[开发日志](docs/开发日志.md)；不得将它们表述为本轮生产桌面 E2E 或完整研究验收。

v0.2.2 解包程序位于 `data/release/desktop-v022/win-unpacked/Paper Agent.exe`，包含已构建的 Python 服务。本机到 GitHub 的 TLS 不可用，`winCodeSign` 下载失败，打包使用本地 Electron 分发并关闭 EXE 资源编辑/签名；应用内 `ServiceInfo.version` 为 0.2.2，EXE 文件版本元数据仍显示 Electron 45.0.0。旧 `data/release/desktop-v021/` 和 `data/release/desktop/` 未覆盖。
