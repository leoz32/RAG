# Dataset Source Note

## Overview

This project uses two different but related data assets:

1. A local textbook corpus stored under `data/raw/`
2. A QA evaluation dataset stored under `data/eval/`

The current main experiment line is:

- textbook corpus: `data/raw/openstax_us_history_clean_v1/`
- QA subset: `data/eval/split_by_source_doc/openstax_qa_dataset.csv`

This means the current evaluation is run on **OpenStax-only questions** against the **cleaned OpenStax U.S. History textbook corpus**.

## QA Dataset Origin

The original QA dataset was downloaded from Hugging Face:

- dataset name: `ambrosfitz/openstax_american_yawp`
- URL: <https://huggingface.co/datasets/ambrosfitz/openstax_american_yawp>

Local source file:

- `data/eval/openstax_american_yawp_qa_dataset.csv`

The dataset contains mixed questions sourced from two textbook families:

- `OpenStax`
- `American Yawp`

Local split summary:

- total mixed QA rows: `7710`
- `OpenStax` rows: `3340`
- `American Yawp` rows: `4370`

The split report is stored at:

- `data/eval/split_by_source_doc/split_report.json`

## Current Evaluation Subset

For the current main experiments, the mixed QA dataset is filtered by:

- `source_doc = OpenStax`

The resulting evaluation file is:

- `data/eval/split_by_source_doc/openstax_qa_dataset.csv`

This subset contains `3340` question rows and is used to reduce source mismatch between the QA set and the retrieval corpus.

## Textbook Corpus Origin

The retrieval corpus used in the current main experiments is:

- `data/raw/openstax_us_history_clean_v1/`

This corpus is a locally prepared markdown version of **OpenStax U.S. History**, derived from the official OpenStax textbook pages and then cleaned for retrieval use.

OpenStax official book site:

- <https://openstax.org/details/books/us-history>

OpenStax table of contents / online pages:

- <https://openstax.org/books/us-history/pages/index>

Local preparation flow:

1. fetch raw OpenStax pages
2. convert/store as markdown
3. clean instructional boilerplate and tail matter
4. build chunks and vector index from the cleaned markdown corpus

## Why The Split Matters

The original Hugging Face QA dataset is not single-source. If the full mixed set is evaluated against only the OpenStax textbook corpus, retrieval quality drops because many questions are derived from `American Yawp` rather than OpenStax.

To reduce this mismatch, the current main experiment uses:

- OpenStax textbook corpus
- OpenStax-only QA subset

## Recommended Citation Note

When describing the data pipeline in documentation or papers, use wording close to:

> The evaluation questions were derived from the Hugging Face dataset `ambrosfitz/openstax_american_yawp`. For the main OpenStax-aligned experiments, we filtered this mixed dataset to rows labeled `source_doc = OpenStax`, yielding 3,340 questions. Retrieval was performed over a locally cleaned markdown corpus of the official OpenStax *U.S. History* textbook.
