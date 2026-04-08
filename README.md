# Rag_Hallucination_eval

面向高教教学问答场景的研究型 RAG 幻觉评估平台。它不是在线问答产品，而是一个可复现实验系统，用于比较 `Non-RAG baseline` 与 `RAG` 两条生成链在同一批教学问题上的幻觉差异，并输出自动化评测结果与论文友好分析报告。

## 研究定位

- `baseline`：只把问题发给 LLM，不注入任何检索上下文。
- `rag`：先从本地 FAISS 知识库检索相关教学资料，再把上下文注入 Prompt。
- 评测目标：比较两条链在 `faithfulness`、`answer_relevance` 等指标上的表现差异。

`baseline` 在生成阶段绝不能调用 retriever。这是实验对照组的基本约束。  
`baseline` 在评测阶段允许补入 `evaluation_contexts`，但该字段只用于评测事实对照，不参与答案生成。

## 当前最小闭环

首版默认：

- Python 3.11+
- 默认使用 SiliconFlow 的 OpenAI-compatible 接口
- 原始教学资料只接收 `.md`
- 本地 CLI + YAML 配置
- 单机运行

默认配置 [base.yaml](/Users/admin/RAG/configs/base.yaml) 当前为：

- LLM: SiliconFlow 上的 `deepseek-ai/DeepSeek-V3`
- LLM `base_url`: `https://api.siliconflow.cn/v1`
- Embedding: SiliconFlow 上的 `BAAI/bge-large-zh-v1.5`
- Embedding `base_url`: `https://api.siliconflow.cn/v1`
- API Key: `SILICONFLOW_API_KEY`

也支持保留本地 embedding，仅把 DeepSeek 作为 LLM。当前仓库附带 [deepseek_local.yaml](/Users/admin/RAG/configs/deepseek_local.yaml)：

- LLM: `deepseek-chat`
- `base_url`: `https://api.deepseek.com`
- Embedding: 本地可复现 `local_hash`

这样可以在没有外部 embedding 服务时跑通最小实验闭环。

嵌入模型也支持独立配置为 OpenAI-compatible 接口，不必和 LLM 共用同一家服务。当前仓库附带 [deepseek_siliconflow_embedding.yaml](/Users/admin/RAG/configs/deepseek_siliconflow_embedding.yaml)：

- LLM: `deepseek-chat`
- LLM `base_url`: `https://api.deepseek.com`
- Embedding: `BAAI/bge-large-zh-v1.5`
- Embedding `base_url`: `https://api.siliconflow.cn/v1`

这适合 LLM 继续走 DeepSeek、嵌入改走硅基流动的场景。

## 目录

```text
configs/
data/
  raw/
  processed/
  eval/
  vector_store/
results/
  runs/
src/
scripts/
```

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

在 `.env` 中填入：

```bash
SILICONFLOW_API_KEY=...
DEEPSEEK_API_KEY=...
OPENAI_API_KEY=...
```

如果你想显式声明默认的 SiliconFlow 方案，配置示例：

```yaml
llm:
  provider: openai_compatible
  model_name: deepseek-ai/DeepSeek-V3
  api_key_env: SILICONFLOW_API_KEY
  base_url: https://api.siliconflow.cn/v1

embedding:
  provider: openai_compatible
  model_name: BAAI/bge-large-zh-v1.5
  api_key_env: SILICONFLOW_API_KEY
  base_url: https://api.siliconflow.cn/v1

evaluation:
  llm_max_tokens: 4096
```

其中：

- `llm.max_tokens` 只控制生成阶段回答长度
- `evaluation.llm_max_tokens` 只控制 Ragas 评测阶段的结构化判分输出长度
- 如果评测日志里出现 `max_tokens length limit`，优先继续增大 `evaluation.llm_max_tokens`

接口调用范式与 OpenAI embeddings 兼容，请求会命中 `/embeddings`，等价于：

```python
import requests

url = "https://api.siliconflow.cn/v1/embeddings"
payload = {
    "model": "BAAI/bge-large-zh-v1.5",
    "input": "Silicon flow embedding online: fast, affordable, and high-quality embedding services.",
}
headers = {
    "Authorization": "Bearer <token>",
    "Content-Type": "application/json",
}
response = requests.post(url, json=payload, headers=headers)
print(response.text)
```

## 数据准备

1. 将课程资料整理为 Markdown 文件，放入 `data/raw/`
2. 准备题集 `data/eval/qa_dataset.csv`

题集最小字段：

```text
question_id,question,ground_truth,question_type
```

推荐 `question_type`：

- `factoid`
- `reasoning`
- `adversarial`

## 运行

默认运行方案使用 [base.yaml](/Users/admin/RAG/configs/base.yaml)，即 SiliconFlow 的 `deepseek-ai/DeepSeek-V3` + `BAAI/bge-large-zh-v1.5`。

### Stage 1: 配置检查

```bash
python -m src.main --config configs/base.yaml show-config
```

验收条件：

- 能加载 `configs/base.yaml`
- 能自动创建 `results/runs/{experiment_id}/`
- 能生成 `config_snapshot.yaml`

### Stage 2: 建索引与生成

```bash
python scripts/build_index.py --config configs/base.yaml
python scripts/run_generation.py --config configs/base.yaml --experiment-id <experiment_id>
```

验收条件：

- `data/processed/chunks.jsonl` 已生成
- `data/vector_store/faiss_index/` 已生成
- `results/runs/<experiment_id>/generation_results.csv` 已生成
- `generation_results.csv` 同时包含 `baseline` 与 `rag`

### Stage 3: 评测与分析

```bash
python scripts/run_evaluation.py --config configs/base.yaml --experiment-id <experiment_id>
```

验收条件：

- `baseline_eval_results.csv` 已生成
- `rag_eval_results.csv` 已生成
- `metrics_summary.json` 已生成
- `analysis_report.json` 已生成

### Stage 4: 一键完整实验

```bash
python scripts/run_full_experiment.py --config configs/base.yaml
```

如果你只使用默认硅基流动方案，最小准备通常只需要：

```bash
source .venv/bin/activate
export SILICONFLOW_API_KEY=...
python -m src.main --config configs/base.yaml show-config
python scripts/run_full_experiment.py --config configs/base.yaml
```

验收条件：

- 三段脚本共享同一个 `experiment_id`
- 所有结果统一落在 `results/runs/<experiment_id>/`

## 论文实验说明

### 为什么同时设置 Non-RAG 和 RAG

这是为了构建清晰对照组。`baseline` 表示模型只依赖参数知识回答；`rag` 表示模型显式依赖外部教学资料回答。两者对比可以直接支撑“RAG 是否降低幻觉”这一研究问题。

### 为什么 baseline 评测时可以有参考 contexts

评测需要事实参照物。为避免把生成阶段上下文和评测阶段上下文混淆，系统单独使用 `evaluation_contexts` 字段。它只在 evaluation 层注入，不会回流到 baseline 生成链。

### 指标含义

- `faithfulness`：答案是否忠实于参考上下文
- `answer_relevance`：答案是否真正回应问题
- `context_precision`：检索上下文是否足够聚焦
- `context_recall`：检索上下文是否覆盖应有信息

## 已实现模块

- 配置与唯一契约源
- Markdown 资料加载、清洗、切块、FAISS 建库
- baseline / rag 两条生成链
- Ragas 评测、汇总统计、分析报告

## 暂未实现

- PDF/TXT/DOCX/PPTX 原生解析
- 多 provider 抽象
- 人工标注工作流
- Notebook 调试文件
