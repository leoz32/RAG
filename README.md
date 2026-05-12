# RAG Hallucination Evaluation for Educational QA

## 1. 项目说明

本项目是一个面向高等教育知识问答场景的研究型 RAG 幻觉评估框架，用于系统比较 `Non-RAG baseline` 与 `RAG` 两条生成链路在相同问题集上的表现差异，并分析检索增强机制对大模型幻觉抑制的实际作用。

项目的核心目标不是构建在线问答产品，而是提供一套可复现、可对比、可量化分析的实验平台，用于回答以下研究问题：

1. 在高教问答场景中，引入外部知识检索后，模型回答的证据支撑性是否能够稳定提升。
2. 检索增强是否能够显著降低无依据扩写、事实性偏差和 unsupported claims。
3. 幻觉降低带来的收益，是否能够进一步转化为更优的最终回答质量。

围绕上述目标，项目实现了从语料准备、知识切分、索引构建、混合检索、双链路生成到自动评测与结果汇总的完整流程。整体上，项目将 `baseline` 与 `RAG` 放在同一基础模型、相同解码参数与相同问题集下进行对照，仅改变生成阶段是否接入外部证据，从而更准确地识别 RAG 机制本身的净效应。

目前项目支持两类主要研究任务：

- 高教知识问答任务：以 OpenStax 教材语料和课程问答数据为基础，分析 RAG 在教学问答中的幻觉抑制效果与教学可用性。
- 事实生成任务：以 FActScore 数据为补充验证场景，分析 RAG 在长文本事实生成任务中的原子事实支撑能力。

## 2. 项目详细架构

项目整体采用“数据准备—检索增强生成—自动评测—结果分析”的分层架构，主要由以下模块组成。

### 2.1 数据与语料层

该层负责教材语料、问答数据和知识源文件的组织与预处理，是后续检索与评测的基础。

- `data/raw/`：存放本地教材语料、清洗后的 Markdown 文本及其他原始知识源。
- `data/eval/`：存放评测题集、拆分后的 QA 数据与辅助评测数据。
- `FActScore/`：本地集成的 FActScore 源码与配套资源，用于事实级自动评测。

在研究流程中，原始教学语料首先经过清洗与标准化处理，再按照设定的分块策略切分为多个语义单元，以便后续建立向量索引和执行混合检索。

### 2.2 配置层

该层用于管理不同实验的参数设置与运行路径，使项目能够在不同数据集、不同检索策略和不同评测设置之间切换。

- `configs/`：存放各实验配置文件。
- `src/config/`：定义配置结构、运行参数加载与实验路径解析逻辑。

配置文件中统一描述模型参数、检索参数、切分参数、评测参数、数据路径和实验输出路径，从而保证实验具有较强的可复现性。

### 2.3 知识建库层

该层负责将原始教材语料转化为可供检索的知识库。

- `src/ingestion/loaders.py`：加载本地教材文件或知识文本。
- `src/ingestion/cleaner.py`：执行文本清洗。
- `src/ingestion/splitter.py`：按设定规则进行文本切分。
- `src/ingestion/embedder.py`：计算文本块的向量表示。
- `src/ingestion/vector_store.py`：构建并管理向量索引。
- `scripts/build_index.py`：建库入口脚本。

该层的输出通常包括：

- 切分后的文本块文件
- 向量化后的索引文件
- 与语料相绑定的元数据

### 2.4 检索层

该层负责根据用户问题从知识库中召回相关证据，是 RAG 链路的关键组成部分。

- `src/retrieval/retriever.py`：统一检索调度入口。
- `src/retrieval/bm25_retriever.py`：BM25 稀疏检索实现。

当前主线实验采用混合检索策略，综合利用：

- 稠密向量检索
- BM25 稀疏检索
- RRF 融合排序

这种设计能够同时兼顾语义相似性匹配与关键词精确匹配，更适合高教问答中常见的术语定位、定义识别与事实核验任务。

### 2.5 生成层

该层实现 `Non-RAG baseline` 与 `RAG` 两条生成链路，并确保二者可以在统一问题集上进行受控对比。

- `src/generation/prompts.py`：定义 baseline 与 RAG 的 Prompt 模板。
- `src/generation/llm_client.py`：统一大模型调用接口。
- `src/generation/baseline_chain.py`：baseline 链路实现。
- `src/generation/rag_chain.py`：RAG 链路实现。
- `scripts/run_generation.py`：生成阶段主入口。

其中：

- `baseline` 仅接收问题本身，不引入任何检索证据；
- `RAG` 在问题前拼接检索到的上下文，并通过显式指令要求模型严格依据资料回答。

两条链路使用相同基础模型和相同生成参数，确保实验比较的核心差异只来自“是否接入外部证据”。

### 2.6 评测层

该层负责对双链路输出进行自动评测，并从链路级与事实级两个角度分析回答质量。

- `src/evaluation/dataset_builder.py`：构建评测输入表。
- `src/evaluation/dataset_source.py`：读取问题集与评测数据源。
- `src/evaluation/ragas_runner.py`：运行 RAGAS 评测。
- `src/evaluation/factscore_runner.py`：运行 FActScore 评测。
- `src/evaluation/metrics_summary.py`：汇总指标结果。
- `scripts/run_evaluation.py`：评测主入口脚本。

该层的主要输出包括：

- 问题级评测结果
- baseline 与 RAG 的总体平均指标
- 配对比较统计结果
- 供人工抽样复核的候选样本

### 2.7 结果分析层

该层在自动评测结果基础上，进一步生成统计汇总与结构化分析报告。

- `src/analysis/aggregator.py`：执行总体统计与配对分析。
- `src/analysis/report_generator.py`：生成实验报告。

该层面向的主要分析任务包括：

- 各项指标的总体均值统计
- baseline 与 RAG 的配对增益比较
- 分题型或分数据集的结果拆解
- 低忠实度与失败样本定位

### 2.8 脚本入口层

项目通过 `scripts/` 提供若干统一入口，用于串联完整实验流程：

- `build_index.py`：建立索引
- `run_generation.py`：运行双链路生成
- `run_evaluation.py`：执行自动评测
- `run_full_experiment.py`：串联完整实验流程

因此，从整体架构上看，本项目可以概括为如下链路：

`教材语料 -> 文本清洗与切分 -> 向量索引构建 -> 混合检索 -> baseline/rag 双链路生成 -> RAGAS/FActScore 评测 -> 结果汇总与分析`

## 3. 参数信息表

下表汇总了当前项目主线实验中较为核心的参数设置与含义。

| 参数类别 | 参数名 | 当前设置/示例 | 含义说明 |
|---|---|---|---|
| 基础模型 | `llm.model_name` | `deepseek-chat` | 用于 baseline 与 RAG 生成的基础大模型。 |
| 生成参数 | `llm.temperature` | `0.0` | 控制生成随机性，设为 0 有助于提高实验可复现性。 |
| 生成参数 | `llm.max_tokens` | `800` | 单次回答允许生成的最大 token 数。 |
| 嵌入模型 | `embedding.model_name` | `BAAI/bge-m3` 或 `local-hash-256` | 用于文本块向量化与问题向量化的嵌入模型。 |
| 检索策略 | `retrieval.strategy` | `hybrid_rrf` | 检索主策略，表示融合稠密检索与 BM25 的混合检索。 |
| 稠密召回数 | `retrieval.dense_top_k` | `5` | 稠密向量检索初步召回的候选数。 |
| 稀疏召回数 | `retrieval.bm25_top_k` | `8` | BM25 检索初步召回的候选数。 |
| 最终返回数 | `retrieval.top_k` | `5` | 融合排序后最终传入生成模型的上下文片段数。 |
| RRF 参数 | `retrieval.rrf_k` | `30` | 倒数排序融合的控制参数。 |
| 切分策略 | `chunking.strategy` | `sentence_window` | 文本切分方式，按句窗口组织语义片段。 |
| 块长度 | `chunking.chunk_size` | `700` | 单个文本块的目标长度上限。 |
| 块重叠 | `chunking.chunk_overlap` | `120` | 相邻文本块之间的重叠长度。 |
| 目标句数 | `chunking.target_sentences_per_chunk` | `5` | 每个文本块尽量包含的目标句数。 |
| 句级重叠 | `chunking.sentence_overlap` | `2` | 相邻块之间共享的句子数。 |
| 最大词数 | `chunking.max_words_per_chunk` | `220` | 单个文本块的词数硬上限。 |
| 并发生成 | `generation.concurrency` | `4` | 生成阶段并发执行的问题数。 |
| RAGAS 评测并发 | `evaluation.concurrency` | `1` | RAGAS 评测时的并发数。 |
| FActScore 并发 | `evaluation.factscore_concurrency` | `4` | FActScore 评测时的并发数。 |
| FActScore 开关 | `evaluation.factscore_enabled` | `true` | 是否启用 FActScore 事实级评测。 |
| FActScore 模型 | `evaluation.factscore_model_name` | `retrieval+ChatGPT` | FActScore 使用的事实支持判别模式。 |
| FActScore 惩罚项 | `evaluation.factscore_gamma` | `10` | FActScore 总分中的长度惩罚参数。 |
| Prompt 版本 | `prompt_version` | 如 `openstax_dataset_export_cleaned_20260422_v1` | 标记本轮实验使用的 Prompt 模板版本。 |
| 数据集类型 | `dataset.kind` | `qa_csv` / `factscore_jsonl` | 标记当前实验使用的是问答型数据还是 FActScore 格式数据。 |
| 数据集版本 | `dataset.dataset_version` | 如 `20260422_v1`、`v1` | 标记具体实验所对应的数据版本。 |
| 语料路径 | `paths.raw_dir` | 本地教材语料目录 | 存放检索知识源原文的位置。 |
| 结果路径 | `paths.results_dir` | `results` | 实验结果与汇总报告输出目录。 |

为方便理解，不同实验线的代表性配置文件如下：

- OpenStax 高教问答主线配置：[configs/openstax_dataset_export_cleaned_20260422.yaml](/home/leoz32/RAG/configs/openstax_dataset_export_cleaned_20260422.yaml)
- FActScore 人物简介事实生成配置：[configs/factscore_unlabeled_chatgpt_local.yaml](/home/leoz32/RAG/configs/factscore_unlabeled_chatgpt_local.yaml)

如果后续需要扩展到新的课程语料、新的嵌入模型或新的评测组合，通常只需在 `configs/` 中新增或修改配置文件，而不必改动整体框架结构。

## 4. 实验启动

本节给出项目的基本启动流程，包括环境配置、数据准备、生成与评测命令，便于复现实验。

### 4.1 环境配置

建议使用独立虚拟环境安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

项目中常用的环境变量如下：

```bash
DEEPSEEK_API_KEY=...
SILICONFLOW_API_KEY=...
OPENAI_API_KEY=...
```

其中：

- `DEEPSEEK_API_KEY`：用于生成阶段和部分评测阶段的大模型调用；
- `SILICONFLOW_API_KEY`：用于嵌入模型调用；
- `OPENAI_API_KEY`：在部分兼容流程或外部依赖中可能需要保留。

### 4.2 数据集准备

#### 4.2.1 OpenStax 问答实验

若需从上游 Hugging Face 数据集适配本地 QA 文件，可运行：

```bash
python scripts/datasets/openstax_american_yawp/prepare_dataset.py
```

若需将混合题集按来源拆分，可运行：

```bash
python scripts/split_qa_dataset_by_source_doc.py \
  --input data/eval/openstax_american_yawp_qa_dataset.csv \
  --output-dir data/eval/split_by_source_doc
```

若需抓取并清洗 OpenStax 教材原文，可运行：

```bash
python scripts/fetch_openstax_us_history.py \
  --i-have-permission \
  --out-dir data/raw/openstax_us_history_original_v3
```

```bash
python scripts/clean_openstax_us_history.py \
  --input-dir data/raw/openstax_us_history_original_v3 \
  --output-dir data/raw/openstax_us_history_clean_v1
```

正式实验也可直接使用本地整理后的配置：

- [configs/openstax_dataset_export_cleaned_20260422.yaml](/home/leoz32/RAG/configs/openstax_dataset_export_cleaned_20260422.yaml)

#### 4.2.2 FActScore 实验

FActScore 相关源码已集成在仓库内。若需导入或整理本地数据，可运行：

```bash
python scripts/import_factscore_dataset.py
```

对应配置文件为：

- [configs/factscore_unlabeled_chatgpt_local.yaml](/home/leoz32/RAG/configs/factscore_unlabeled_chatgpt_local.yaml)

### 4.3 建立索引

在 OpenStax 主线实验中，首先需要对教材语料建立索引：

```bash
python scripts/build_index.py \
  --config configs/openstax_dataset_export_cleaned_20260422.yaml \
  --experiment-id openstax_run
```

### 4.4 运行生成

小样本试运行：

```bash
python scripts/run_generation.py \
  --config configs/openstax_dataset_export_cleaned_20260422.yaml \
  --experiment-id openstax_run \
  --limit 100
```

全量生成：

```bash
python scripts/run_generation.py \
  --config configs/openstax_dataset_export_cleaned_20260422.yaml \
  --experiment-id openstax_run
```

若运行 FActScore 主线实验，可将配置替换为：

```bash
python scripts/run_generation.py \
  --config configs/factscore_unlabeled_chatgpt_local.yaml \
  --experiment-id factscore_run
```

### 4.5 运行评测

OpenStax 主线评测：

```bash
python scripts/run_evaluation.py \
  --config configs/openstax_dataset_export_cleaned_20260422.yaml \
  --experiment-id openstax_run
```

FActScore 主线评测：

```bash
python scripts/run_evaluation.py \
  --config configs/factscore_unlabeled_chatgpt_local.yaml \
  --experiment-id factscore_run
```

### 4.6 一键运行完整流程

如果希望串联完整实验流程，可运行：

```bash
python scripts/run_full_experiment.py \
  --config configs/openstax_dataset_export_cleaned_20260422.yaml \
  --experiment-id openstax_run
```
