# Rag Hallucination Eval

面向高教教学问答场景的研究型 RAG 幻觉评估平台。用于比较 `Non-RAG baseline` 与 `RAG` 两条生成链在同一批教学问题上的幻觉差异。

`教材准备 -> 建库 -> 检索 -> baseline / rag 生成 -> Ragas 评测`

## 当前主线

当前推荐实验路径是：

- 语料：`OpenStax U.S. History` 清洗后的 Markdown
- 题集：只保留 `source_doc = OpenStax` 的 QA 子集
- 检索：`bge-m3` + FAISS dense + BM25 + `hybrid_rrf`
- 生成：`baseline` 与 `rag` 两条链并行输出
- 评测：Ragas，支持断点续跑

项目里仍保留早期的 `openstax_american_yawp` 适配器和纯 dense 路径，但它们不再是默认推荐方案。

## 目录

```text
configs/
data/
  raw/
  eval/
  processed/
  vector_store/
results/
  runs/
scripts/
src/
```

说明：

- `data/raw/`、`data/eval/` 下的抓取教材和实验题集默认只在本地保留，不纳入版本控制。
- `data/processed/`、`data/vector_store/`、`results/runs/` 都是运行产物，不提交。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

常用环境变量：

```bash
SILICONFLOW_API_KEY=...
DEEPSEEK_API_KEY=...
OPENAI_API_KEY=...
```

## 当前关键配置

推荐配置：

- `configs/openstax_us_history_clean_openstax_only.yaml`

它使用：

- LLM: `deepseek-chat`
- Embedding: `BAAI/bge-m3`
- Dense retrieval + BM25 hybrid RRF
- OpenStax-only QA 子集

辅助配置：

- `configs/openstax_us_history_clean.yaml`
  - 用清洗后的 OpenStax 原文，但题集仍指向完整混合 QA
- `configs/openstax_american_yawp_local.yaml`
  - 面向上游 HF 数据集适配后的本地 smoke test 路线

## 数据准备

### 1. 从 Hugging Face 适配 QA 数据

```bash
python scripts/datasets/openstax_american_yawp/prepare_dataset.py
```

该脚本会把 `ambrosfitz/openstax_american_yawp` 映射成项目内部格式。注意：

- 上游数据集是 QA 数据，不是教材原文语料
- `data/raw/openstax_american_yawp/**/*.md` 是派生 pseudo-corpus，只适合 smoke test，不适合作为正式实验的检索语料

### 2. 抓取并清洗 OpenStax U.S. History 原文

先抓取：

```bash
python scripts/fetch_openstax_us_history.py \
  --i-have-permission \
  --out-dir data/raw/openstax_us_history_original_v3
```

再清洗：

```bash
python scripts/clean_openstax_us_history.py \
  --input-dir data/raw/openstax_us_history_original_v3 \
  --output-dir data/raw/openstax_us_history_clean_v1
```

### 3. 按 `source_doc` 拆分题集

```bash
python scripts/split_qa_dataset_by_source_doc.py \
  --input data/eval/openstax_american_yawp_qa_dataset.csv \
  --output-dir data/eval/split_by_source_doc
```

会生成：

- `data/eval/split_by_source_doc/openstax_qa_dataset.csv`
- `data/eval/split_by_source_doc/american_yawp_qa_dataset.csv`

## 运行

### 1. 建索引

```bash
python scripts/build_index.py \
  --config configs/openstax_us_history_clean_openstax_only.yaml \
  --experiment-id exp_openstax_only_bgem3
```

### 2. 小样本生成

先跑 100 题，而不是直接跑全量：

```bash
python scripts/run_generation.py \
  --config configs/openstax_us_history_clean_openstax_only.yaml \
  --experiment-id exp_openstax_only_bgem3 \
  --limit 100
```

### 3. 评测

```bash
python scripts/run_evaluation.py \
  --config configs/openstax_us_history_clean_openstax_only.yaml \
  --experiment-id exp_openstax_only_bgem3
```

## 当前实现重点

- 数据集隔离的 `chunks` / `vector_store` 目录
- 索引 metadata 校验，避免不同实验串库
- embedding cache metadata 校验，避免更换 provider / model 后误复用旧缓存
- SiliconFlow embedding 超长输入 fallback
- `sentence_window` 分块 + 词数硬上限
- `run_generation --limit`
- Ragas 断点续跑
- BM25 + dense `hybrid_rrf`

## 注意事项

- `baseline` 在生成阶段不能调用 retriever。
- `evaluation_contexts` 只用于评测，不参与 baseline 生成。
- 调整 `top_k`、`bm25_top_k`、`rrf_k` 不应要求重建索引；重建索引只在语料、embedding、chunking 变化时需要。
- 正式全量实验前，先做检索抽样和 100 题小样本生成。

## 非目标

- 不做在线服务或前端
- 不提交大体积教材原文、索引和实验结果
- 不把 pseudo-corpus 当正式检索语料
