<div align="center">

<p>
  <a href="README.md"><kbd>English</kbd></a>
  <kbd>简体中文 · 当前</kbd>
</p>

<br>

# Alpha Forge

<p>
  <strong>面向系统化交易的 Agentic AI 基础设施。</strong>
</p>

<p>
  Alpha Forge 将 LLM 原生研究、representation learning、组合治理和
  IBKR-ready execution 收束进一个安静、可审计的 operator system。
</p>

<p>
  <a href="https://github.com/Liu-Ming-Yu/alpha-forge/stargazers">
    <img alt="GitHub stars" src="https://img.shields.io/github/stars/Liu-Ming-Yu/alpha-forge?style=flat&amp;logo=github&amp;label=Stars&amp;labelColor=0b0b0f&amp;color=111827">
  </a>
  <a href="https://github.com/Liu-Ming-Yu/alpha-forge/network/members">
    <img alt="GitHub forks" src="https://img.shields.io/github/forks/Liu-Ming-Yu/alpha-forge?style=flat&amp;logo=github&amp;label=Forks&amp;labelColor=0b0b0f&amp;color=111827">
  </a>
  <a href="https://github.com/Liu-Ming-Yu/alpha-forge/issues">
    <img alt="GitHub issues" src="https://img.shields.io/github/issues/Liu-Ming-Yu/alpha-forge?style=flat&amp;logo=github&amp;label=Issues&amp;labelColor=0b0b0f&amp;color=111827">
  </a>
  <a href="https://github.com/Liu-Ming-Yu/alpha-forge/commits/main">
    <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/Liu-Ming-Yu/alpha-forge?style=flat&amp;logo=github&amp;label=Last%20commit&amp;labelColor=0b0b0f&amp;color=111827">
  </a>
  <a href="LICENSE">
    <img alt="GitHub license" src="https://img.shields.io/github/license/Liu-Ming-Yu/alpha-forge?style=flat&amp;label=License&amp;labelColor=0b0b0f&amp;color=111827">
  </a>
</p>

<p>
  <img alt="Python 3.11" src="https://img.shields.io/badge/Python-3.11-111827?style=flat&amp;logo=python&amp;logoColor=white&amp;labelColor=0b0b0f">
  <img alt="Docker ready" src="https://img.shields.io/badge/Docker-ready-111827?style=flat&amp;logo=docker&amp;logoColor=white&amp;labelColor=0b0b0f">
  <img alt="IBKR ready" src="https://img.shields.io/badge/IBKR-ready-111827?style=flat&amp;labelColor=0b0b0f">
  <img alt="FastAPI operator API" src="https://img.shields.io/badge/FastAPI-operator%20API-111827?style=flat&amp;logo=fastapi&amp;logoColor=white&amp;labelColor=0b0b0f">
</p>

<p>
  <sub>Agentic LLM intelligence / representation learning / autonomous research / governed execution</sub>
</p>

<p>
  <a href="#demo"><kbd>Demo</kbd></a>
  <a href="#command-deck"><kbd>指挥台</kbd></a>
  <a href="#architecture-map"><kbd>架构图</kbd></a>
  <a href="#agentic-llm-intelligence-layer"><kbd>LLM 智能层</kbd></a>
  <a href="#autonomous-research-factory"><kbd>研究工厂</kbd></a>
  <a href="#production-execution-fortress"><kbd>执行堡垒</kbd></a>
</p>

</div>

---

<a id="demo"></a>

## Demo

<div align="center">

<img src="docs/assets/alpha-forge-console-demo.gif" alt="Alpha Forge operator console 演示，展示实时 NAV、broker 状态、readiness 和策略控制" width="100%">

<br>

<sub>用于查看 NAV、broker health、regime state、strategy readiness 和 execution controls 的实时 operator console。</sub>

</div>

---

<a id="command-deck"></a>

## 指挥台

<table>
  <tr>
    <td width="33%">
      <strong>Discover / 发现</strong><br>
      <sub>浏览产品表层能力。</sub><br><br>
      <a href="#showcase">Showcase</a><br>
      <a href="#architecture-map">架构图</a><br>
      <a href="#technology-stack">技术栈</a>
    </td>
    <td width="33%">
      <strong>Investigate / 深入</strong><br>
      <sub>打开核心智能系统。</sub><br><br>
      <a href="#agentic-llm-intelligence-layer">Agentic LLM Layer</a><br>
      <a href="#text-event-v2">Text-Event-v2</a><br>
      <a href="#hybrid-ml--representation-learning-engine">Hybrid ML Engine</a>
    </td>
    <td width="33%">
      <strong>Operate / 运行</strong><br>
      <sub>理解部署、治理和安全执行。</sub><br><br>
      <a href="#v2-shared-account-orchestrator">V2 Orchestrator</a><br>
      <a href="#production-execution-fortress">Execution Fortress</a><br>
      <a href="#governance--observability-fabric">Governance Fabric</a>
    </td>
  </tr>
</table>

<details open>
<summary><strong>打开 Mission Control</strong> &mdash; Alpha Forge 一眼看懂</summary>

| Mission | System Response |
| --- | --- |
| 将语言转化为 alpha | Agentic LLM layer 将叙事转化为 text-event-v2 features |
| 阻止失控的 AI 行为 | Manifests、evals、startup assertions、audits 和 human-review gates |
| 搜索完整特征空间 | Campaigns、walk-forward validation、IC diagnostics 和 bootstrap evidence |
| 安全合并多策略 | V2 账户级编排器把多个 proposal 解析为一个 portfolio target |
| 阻止不安全执行 | Fail-closed broker path、reconciliation、kill switches 和 durable journals |

</details>

<details>
<summary><strong>打开 Alpha Lifecycle</strong> &mdash; 从原始信号到受控实盘</summary>

```text
Language + Market Data
  -> Agentic Intelligence
  -> Feature Families
  -> Campaign Research
  -> Walk-Forward Validation
  -> Evidence Package
  -> Production-Candidate Gate
  -> Shadow Mode
  -> Paper Trading
  -> Paper Soak
  -> Live Preflight
  -> Controlled Live
```

</details>

<a id="showcase"></a>

## Showcase

| System | Signature Capability |
| --- | --- |
| Agentic LLM Intelligence Layer | 使用 tool use、manifests、traceable reasoning、evals、live assertions 和 human-review gates 的多智能体 text-event-v2 alpha 生成 |
| Hybrid ML + Representation Learning Engine | XGBoost、learned embeddings、PCA representations、formulaic alpha mining、microstructure、ownership、estimates、options、macro 和 text-event intelligence |
| Autonomous Research Factory | Campaign-driven alpha discovery、walk-forward validation、IC diagnostics、bootstrap evidence、regime testing 和 production-candidate packaging |
| V2 Shared-Account Orchestrator | 多引擎 portfolio target fusion，并通过受治理的 Shadow &rarr; Paper &rarr; Live 晋升路径 |
| Production Execution Fortress | Fail-closed broker execution、durable state、reconciliation、kill switches、strict service boundaries 和 automated verification |
| Governance & Observability Fabric | Manifest registry、startup assertions、model provenance、event journals、readiness reports、promotion gates 和 operator visibility |

<a id="architecture-map"></a>

## 架构图

![Alpha Forge 架构海报](docs/assets/alpha-forge-architecture.zh-CN.svg)

<details open>
<summary><strong>探索架构层</strong> &mdash; 进入系统栈</summary>

| Layer | What It Controls | Why It Matters |
| --- | --- | --- |
| Language Intelligence | Agents、text-event-v2 extraction、manifests、evals、assertions | 将市场叙事转化为受治理的 alpha inputs |
| Research Graph | Feature families、campaigns、walk-forward validation、diagnostics | 将假设转化为可复现证据 |
| Portfolio Fusion | Multi-engine proposals、account budgets、risk overlays | 防止不同策略模块争抢同一个账户 |
| Execution Safety | Pre-trade checks、broker adapters、kill switches、reconciliation | 在订单到达券商之前阻止不安全行为 |
| Governance Fabric | Readiness reports、promotion gates、event journals、operator visibility | 让 AI-native trading 可审计、可控制 |

</details>

<details>
<summary><strong>打开 Service Boundary Console</strong> &mdash; 模块化单体如何保持纪律</summary>

| Boundary | Rule |
| --- | --- |
| `core` | 拥有 contracts、domain models 和 events |
| `application` | 拥有 use cases 和 read models |
| `services` | 在服务边界后拥有领域逻辑 |
| `infrastructure` | 拥有持久化 adapters 和外部系统 |
| `bootstrap` | 组合具体 runtime dependencies |
| `cli` / `views` | 保持轻薄，并面向 operator |

</details>

---

<a id="agentic-llm-intelligence-layer"></a>

## Agentic LLM Intelligence Layer

### From language to governed alpha.

平台将语言视为一等市场数据基底。

Filings、earnings calls、macro commentary、news、transcripts、social narratives、
central-bank language、analyst revisions 和 event-driven text 会被转换成结构化的
text-event-v2 features。这些 features 不是不受控制的 LLM outputs。它们绑定 manifest、
被版本化、被评估、被审计，并且在影响 portfolio 之前必须通过 promotion gate。

中心是一层受治理的 multi-agent intelligence layer：一组专门化 AI agents 会围绕市场上下文推理、
挑战假设、检查风险，并为 quantitative validation 准备 candidate signals。

<details open>
<summary><strong>打开 Agent Console</strong> &mdash; 专门化智能角色</summary>

| Agent | Role |
| --- | --- |
| Market Oracle | 将 market events、regime changes 和 narrative shifts 转化为结构化 research hypotheses |
| Sentiment Synthesizer | 从 noisy language streams 中提取 sentiment、uncertainty、surprise 和 directional pressure |
| Strategy Debater | 在 candidate alpha ideas 进入正式 research 前进行 adversarial critique |
| Risk Guardian | 检测 overfitting、regime fragility、crowding、stale data 和 hidden exposure |
| Evidence Auditor | 检查 signal 是否具备足够的统计与运营证据来进入 promotion |
| Execution Examiner | 审查 strategy 是否能承受 cash、liquidity、broker 和 order-routing constraints |

</details>

这不是 &ldquo;prompt engineering&rdquo;。

这是面向 alpha research 的受治理机器推理：agent outputs 可追踪、可复现、可检查，并连接到下游证据。

<a id="text-event-v2"></a>

## Text-Event-v2

### LLM-native alpha, engineered for production.

text-event-v2 family 将非结构化语言转化为 quantitative features，使它们可以像任何 alpha source
一样被排名、回测、压力测试和治理。

它支持：

<details open>
<summary><strong>打开 Text-Event Capability Matrix</strong> &mdash; 治理下的 LLM-native features</summary>

| Capability | Description |
| --- | --- |
| Event Extraction | 将 raw text 转化为 structured market events、catalysts、risks 和 directional hypotheses |
| Narrative Regime Detection | 追踪 market language、macro tone、risk appetite 和 sector-level narratives 的变化 |
| Filing Intelligence | 提取 management tone、uncertainty、litigation risk、guidance changes 和 operating pressure |
| Earnings Call Reasoning | 识别 surprise、confidence、hedging language 和 forward-looking signal changes |
| Manifest-Governed Features | 每个 LLM-derived feature 都绑定 schema、prompt contract、model version 和 audit trail |
| Live Startup Assertions | 如果 text models、manifests 或 required feature contracts 缺失或不一致，生产启动会失败 |
| Promotion Readiness | Text-derived signals 必须通过与 price、fundamental 或 microstructure features 相同的 evidence pipeline |

</details>

结果是一层更像受治理研究仪器、而不是 chatbot 的语言智能系统。

---

<a id="hybrid-ml--representation-learning-engine"></a>

## Hybrid ML + Representation Learning Engine

### Classical alpha, modern ML, and learned structure &mdash; in one research graph.

研究引擎将传统 quantitative signals 与 machine learning、learned representations 结合，
覆盖完整 feature universe。

它集成：

<details open>
<summary><strong>打开 Feature Universe</strong> &mdash; 显式领域知识加上 learned structure</summary>

| Feature / Model Family | Purpose |
| --- | --- |
| Price & Momentum Features | Trend、reversal、volatility、distance-to-high 和 cross-sectional ranking signals |
| Microstructure-v3 | Liquidity、spread、volume pressure、intraday behavior 和 trading-friction-aware features |
| Ownership-v1 | Institutional behavior、positioning shifts、insider activity 和 ownership structure |
| Estimates-v1 | Analyst expectations、revisions、dispersion 和 forward-looking earnings pressure |
| Options-v1 | Volatility surface、skew、flow 和 implied market expectation signals |
| Macro-v1 | Rate、inflation、liquidity、sector 和 broad regime context |
| Formulaic Alpha Mining | 基于 WorldQuant-style grammars 的 evolutionary search，并带有 auto-promotion constraints |
| Learned Representations-v1 | 从 existing features 中抽取 PCA-style embeddings 和 compressed cross-family structure |
| Text-Event-v2 | 来自 filings、calls、news 和 narrative sources 的 LLM-derived structured features |
| XGBoost Ranking Pipelines | Nonlinear interaction discovery、cross-sectional scoring 和 feature attribution |

</details>

这创造出一个可以同时从显式领域知识和 feature families 中隐藏的 latent structure 发现 alpha 的研究系统。

平台不依赖单个 model、单个 signal 或单个 narrative。它构建的是 evidence ensemble。

---

<a id="autonomous-research-factory"></a>

## Autonomous Research Factory

### Alpha discovery with memory, discipline, and promotion control.

Alpha Forge 被设计成一个 autonomous research factory：生成 hypotheses、构建 features、运行 experiments、
评估 results、保存 artifacts，并且只 promotion 那些真正存活下来的东西。

研究循环是结构化的：

```text
Hypothesis -> Feature Family -> Campaign -> Walk-Forward Validation -> Diagnostics -> Evidence Package -> Promotion Decision
```

每个 experiment 都会产生可复现 artifacts：feature manifests、model configs、fold results、
IC diagnostics、bootstrap checks、turnover profiles、drawdown curves 和 production-candidate reports。

<details open>
<summary><strong>打开 Research Diagnostics</strong> &mdash; 如何拒绝漂亮但脆弱的幻觉</summary>

| Research Capability | Description |
| --- | --- |
| Campaign-Based Research | Experiments 以 campaigns 组织，而不是临时 notebooks |
| Walk-Forward Validation | Strategies 在 rolling out-of-sample folds 上测试 |
| IC Diagnostics | 通过 rank correlation、stability 和 negative-streak behavior 衡量 signal quality |
| Bootstrap Evidence | 在 promotion 前衡量统计不确定性 |
| Turnover Analysis | Execution drag 和 capacity pressure 被视为一等约束 |
| Regime Testing | 分析 risk-on、risk-off、volatile 和 low-liquidity periods 中的表现 |
| Production-Candidate Reports | 用部署所需证据打包 strategies |

</details>

系统被设计用来拒绝诱人的幻觉：高收益回测、脆弱 edge、意外 leakage、不稳定 IC，
以及只在成本前看起来漂亮的 strategies。

---

<a id="v2-shared-account-orchestrator"></a>

## V2 Shared-Account Orchestrator

### Many engines. One account. One coherent portfolio.

真实交易不是一组互不相干的 strategy demos。多个 engines 会竞争同一份 cash、risk budget、
exposure limits 和 broker connection。

V2 Shared-Account Orchestrator 通过把独立 strategy proposals 合并成一个 account-level portfolio target
来解决这个问题。

Strategy engines 可以包括：

<details open>
<summary><strong>打开 Engine Rack</strong> &mdash; 一个账户级 target 的 proposal sources</summary>

| Engine | Role |
| --- | --- |
| Cross-Sectional Equity Ranker | 基于 alpha scores 选择并加权 securities |
| ETF Macro Allocator | 基于 macro 和 regime signals 调整 broad exposure |
| Risk Overlay Engine | 当 drawdown、volatility 或 signal quality 恶化时降低 exposure |
| Text-Event Engine | 将 LLM-derived event signals 纳入 portfolio intent |
| Future Strategy Modules | 未来 engines 可以通过同一个 governed interface 提交 proposals |

</details>

在任何 target 可执行之前，orchestrator 会应用 cash constraints、exposure limits、stale-state checks、
throttles、kill switches、reconciliation 和 broker-readiness checks。

部署是分阶段的：

```text
Shadow Mode -> Paper Trading -> Paper Soak -> Live Preflight -> Controlled Live
```

Nothing becomes live by accident.
Nothing bypasses governance.
Nothing trades without a validated account-level view.

---

<a id="production-execution-fortress"></a>

## Production Execution Fortress

### Designed to stop before it fails.

执行层围绕一个保守原则构建：

如果系统无法证明行动是安全的，它就不会行动。

Orders 会经过 fail-closed execution path，包括 broker abstraction、pre-trade validation、
state reconciliation、event journaling 和 promotion-state checks。

<details open>
<summary><strong>打开 Safety Matrix</strong> &mdash; 先证明，再行动</summary>

| Safeguard | Purpose |
| --- | --- |
| Fail-Closed Execution | 当 state stale、invalid、incomplete 或 unverifiable 时阻止行动 |
| Durable Portfolio State | 在重启之间持久化 snapshots、parent-child provenance 和 account state |
| Event Journal | 记录 execution events、promotion events 和 operational state transitions |
| Broker Reconciliation | 比较 intended portfolio state 与 broker/account reality |
| Kill Switches | 当 operational 或 risk conditions 不安全时停止交易 |
| Order Throttling | 防止 runaway order submission 和 broker abuse |
| Paper/Live Separation | 明确隔离 simulation、paper 和 live paths |
| IBKR Gateway/TWS Integration | 通过受控 adapters 支持 broker-backed paper 和 live execution |

</details>

系统不是为了显得活跃而优化。
它是为了正确、可观察、可控制而优化。

---

<a id="governance--observability-fabric"></a>

## Governance & Observability Fabric

### The control plane for AI-native trading.

现代 AI systems 需要可观测、可测试、可治理，尤其是 agents 使用 tools 并与外部系统交互时。
OpenAI 的 [Agents](https://platform.openai.com/docs/guides/agents)、
[Agents SDK tracing](https://openai.github.io/openai-agents-python/tracing/)
和 [human-in-the-loop](https://openai.github.io/openai-agents-python/human_in_the_loop/)
文档直接强调 tools、integrations、observability、guardrails 和 human review 是生产系统核心概念。

Alpha Forge 将这些理念应用到 systematic trading。

<details open>
<summary><strong>打开 Governance Console</strong> &mdash; AI-native trading 的控制平面</summary>

| Governance Layer | Function |
| --- | --- |
| Feature Registry | 跟踪所有 feature families 及其 schemas |
| Model Manifest System | 记录 model versions、feature contracts、prompts、configs 和 artifacts |
| Startup Assertions | 当 required contracts 缺失时阻止生产启动 |
| Readiness Reports | 总结 strategy 是否具备 paper 或 live progression 资格 |
| Promotion Gates | 阻止未通过 evidence、risk 或 operational requirements 的 strategies |
| Traceable Agent Actions | 让 LLM/agent behavior 可审计，而不是 opaque |
| Human Review Hooks | 允许 sensitive transitions 需要显式 approval |
| Operator API | 提供对 system state 和 readiness 的受控可见性 |

</details>

平台不会因为 AI output 听起来聪明就信任它。
它只信任可以被测量、重放、审计和治理的东西。

---

<a id="technology-stack"></a>

## Technology Stack

<details open>
<summary><strong>打开 Stack Matrix</strong> &mdash; Alpha Forge 背后的生产组件</summary>

| Area | Stack |
| --- | --- |
| Language & Tooling | Python 3.11、uv、pyproject.toml、pre-commit、ruff、mypy、pytest |
| ML / Representation Learning | XGBoost、learned representations、PCA embeddings、feature attribution、rank modeling |
| LLM / Agentic AI | Text-event-v2 pipelines、manifest-governed agents、tool-using research workflows、audit trails |
| Data & Research Storage | Parquet、object-store-ready artifacts、campaign outputs、reproducible experiment persistence |
| State & Infrastructure | PostgreSQL、Alembic、Redis/event-bus patterns、durable event journals |
| Execution | Simulated backend、paper backend、IBKR ibapi、controlled live-adapter pathway |
| Deployment | Docker、docker-compose、Makefile orchestration、FastAPI operator API |
| Verification | make verify、import-boundary checks、module-size guards、type-debt enforcement、regression tests |
| Governance | Readiness reports、production-candidate gates、startup assertions、promotion evidence、paper-soak validation |

</details>

---

## Operating Surface

安装后的 console script 与 module entrypoint 等价：

```bash
quant-platform --help
python -m quant_platform --help
```

## Setup

项目验证使用 Python 3.11。

### Quick start（推荐）

一个命令会创建 venv、安装依赖、安装 `ibapi`（IBKR TWS API -- 不在 PyPI
上，会自动从 IBKR 下载），并从 example 初始化 `.env`：

```powershell
pwsh scripts/setup.ps1            # 添加 -Extras 可安装 ml + backtest research extras
```

macOS/Linux/WSL：

```bash
bash scripts/setup.sh             # 添加 --extras 可安装 ml + backtest
```

然后编辑 `.env`（设置 `POSTGRES_PASSWORD`、`QP__API__OPERATOR_API_KEY`，
以及你的 vendor/broker keys）并启动平台 -- 经纪商能力 API 原生运行
（`scripts/serve_api_native.ps1`），或启动完整 Docker stack
（`scripts/deploy.ps1`）。下面是每一步的手动等价命令。

### Manual setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,api]"
```

macOS/Linux/WSL：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,api]"
```

可选 extras：

```bash
python -m pip install -e ".[ml]"       # XGBoost research
python -m pip install -e ".[backtest]" # vectorbt backtests
```

`ibapi`（IBKR TWS API）不在 PyPI 上，所以没有 pip extra 能提供它。
`scripts/setup.ps1` 会自动安装；如需单独（重新）安装：

```bash
pwsh scripts/install_ibapi.ps1            # pinned to API 10.46.1; -Version <ver> to override
bash scripts/install_ibapi.sh             # IBAPI_VERSION=<ver> to override
```

IBKR 会轮换 hosted builds。如果 pinned download 404，请从
<https://interactivebrokers.github.io/> 传入当前版本（例如
`-Version 1047.01`）。

learned-representations 和 sequence-ranker 路径的可选 GPU acceleration 会通过
`[dl]` extra 安装 torch。PyPI 发布的是 CPU-only wheels；随后安装与你的 GPU 匹配的
CUDA build（示例：CUDA 12.8 / cu128）：

```bash
python -m pip install -e ".[dl]"
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

GPU 是可选项。没有 CUDA 时，依赖 torch 的测试会自动跳过。

## Run With Docker

Durable paper/live state 保存在 Postgres 和 Redis 中，它们以 Docker services
运行。完整 stack 可以用一个命令启动：构建 image、启动 Postgres 和 Redis、应用
migrations、启动 operator API 和 console。deploy script 第一次运行时会把缺失的
secrets 生成到 `.env`：

```powershell
pwsh scripts/deploy.ps1            # 添加 -Workers 启动 maintenance worker，-Paper 启动 paper engine，-Rebuild 强制重建
```

macOS/Linux/WSL：

```bash
bash scripts/deploy.sh
```

如果想在 venv 中原生运行平台，但仍使用 durable state，只启动 backing services
并让 `.env` 指向它们即可。`POSTGRES_PASSWORD` 必须先写入 `.env`，因为 compose
会读取它：

```bash
docker compose up -d postgres redis
```

## Live IBKR Broker (Native API)

Docker image 有意不包含 `ibapi`，所以 containerized API 无法连接 IBKR。要拉取
live positions/NAV，请在你的 venv 中运行 operator API（此处已安装 `ibapi`），并在
`127.0.0.1` 上监听 -- TWS API 默认信任该地址，而不会默认信任 container bridge
address -- 同时继续使用 Dockerized Postgres + Redis：

```bash
pwsh scripts/serve_api_native.ps1
bash scripts/serve_api_native.sh
```

该脚本会确保 `ibapi` 存在，启动 Postgres + Redis，停止 Dockerized API 以释放端口，
然后原生启动服务。TWS 或 IB Gateway 必须正在运行，并且按 `.env` 中的
`QP__BROKER__*` 设置可访问（paper TWS = `7497`）。

脚本还会把 `.env` 里的 `QP__LIVE_IBKR__CONTRACTS_FILE` 导出到 process
environment，让 broker-sync 能映射账户持仓（它直接从 `os.environ` 读取该变量，
不是通过 pydantic settings）-- 请指向你的交易 universe，例如
`infra/config/universe_300.json`，否则 console 会显示 0 positions。operator API
启动时会从最新持久化的 broker snapshot hydrate cash/NAV，所以 console 会反映真实账户，
而不是 synthetic `--initial-cash` ledger。

## Configuration

`PlatformSettings` 会从 `.env` 和带 `QP__` 前缀的环境变量加载配置。

从这里开始：

```bash
copy infra\config\settings.example.env .env
```

最小本地 in-memory 开发：

```bash
set QP__STORAGE__POSTGRES_DSN=
set QP__STORAGE__REDIS_URL=
set QP__STORAGE__EVENT_BUS_BACKEND=in_memory
set QP__BROKER__PAPER_TRADING=true
```

Durable paper/live 至少需要：

```bash
QP__STORAGE__POSTGRES_DSN=postgresql+psycopg://user:pass@host:5432/quant_platform
QP__STORAGE__REDIS_URL=redis://localhost:6379/0
QP__STORAGE__EVENT_BUS_BACKEND=redis_streams
QP__API__OPERATOR_API_KEY=<strong random key>
```

operator console 的 Strategy screen（strategy runs、engine budgets、source
contributions）由 V2 unified runtime 填充。启用后，single-engine supervise 会作为
N=1 的 multi-engine case 运行，并写入这些 rows（ADR-014）；`run-multi-engine` 也需要它：

```bash
QP__V2__ENABLED=true
QP__V2__ACCOUNT_ORCHESTRATOR_ENABLED=true
```

常见 IBKR ports：

| App | Mode | Port |
| --- | --- | ---: |
| TWS | Paper | `7497` |
| TWS | Live | `7496` |
| IB Gateway | Paper | `4002` |
| IB Gateway | Live | `4001` |

## Common Commands

Schema：

```bash
python -m quant_platform migrations-check
python -m quant_platform migrate
python -m quant_platform verify-schema
```

单次 paper cycle：

```bash
python -m quant_platform run-cycle --initial-cash 50000
```

Bounded engine runs：

```bash
python -m quant_platform run-engine --mode shadow --cycles 5
python -m quant_platform run-engine --mode paper --execution-backend ib-paper --contracts-file infra/config/paper_contracts.json --cycles 1
python -m quant_platform run-engine --mode live --contracts-file ./contracts.json --cycles 1
```

V2 multi-engine proposal/orchestration path：

```bash
python -m quant_platform run-multi-engine ^
  --engines cross_sectional_equity,etf_macro_allocator ^
  --budgets-file ./budgets.json ^
  --mode paper ^
  --contracts-file infra/config/paper_contracts.json ^
  --cycles 1
```

Data and research：

```bash
python -m quant_platform ingest --start YYYY-MM-DD --end YYYY-MM-DD --contracts-file ./contracts.json
python -m quant_platform maintain --interval 900 --contracts-file ./contracts.json
python -m quant_platform compute-features --contracts-file ./contracts.json
python -m quant_platform features backfill --contracts-file ./contracts.json --start YYYY-MM-DDT00:00:00+00:00 --end YYYY-MM-DDT00:00:00+00:00 --feature-set-version paper-alpha-composite-v1 --date-policy nyse-sessions
python -m quant_platform boosting gpu-check
python -m quant_platform research-campaign run --help
```

Operator API：

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
set QP__API__OPERATOR_API_KEY=<generated key>
python -m quant_platform serve-api --host 127.0.0.1 --port 8000
curl -H "X-API-Key: %QP__API__OPERATOR_API_KEY%" http://127.0.0.1:8000/health/ready
```

## Verification

快速本地 checks：

```bash
python scripts/check_import_boundaries.py
python scripts/check_service_coupling.py
python scripts/check_module_size.py
python scripts/check_type_debt.py
python scripts/check_lint_debt.py --skip-ruff-probe
python -m pytest -q tests/unit/test_engine_loop.py
```

完整 offline gate：

```bash
make verify
```

Durable 和 live gates 需要显式 opt-in：

```bash
set QP_VERIFY_DURABLE=1
set QP_VERIFY_LIVE_IBKR=1
set IBAPI_PACKAGE_PATH=<path-to-TWS-API/source/pythonclient>
make verify
```

## Documentation Map

从 [USEME.md](USEME.md) 开始查看 operator commands，
从 [CONTEXT.md](CONTEXT.md) 开始查看 project vocabulary。

Architecture：

- [System overview](docs/architecture/system-overview.md)
- [Service boundaries](docs/architecture/service-boundaries.md)
- [Production roadmap](docs/architecture/production-roadmap.md)
- [Risk register](docs/architecture/risk-register.md)
- [Source audit checklist](docs/architecture/source-audit-checklist.md)
- [V2 execution flow](docs/architecture/v2-execution-flow.md)
- [Core contracts](docs/interfaces/core-contracts.md)

Runbooks 位于 [docs/runbooks](docs/runbooks/) 下，用于 operations、incident response、
paper promotion、backups、IBKR recovery、data recovery 和 production readiness。
