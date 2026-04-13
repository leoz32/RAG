## OpenStax American Yawp Adapter

This folder contains the adapter for [`ambrosfitz/openstax_american_yawp`](https://huggingface.co/datasets/ambrosfitz/openstax_american_yawp).

The upstream dataset is a QA dataset, not a raw textbook corpus. That creates one important limitation:

- `data/eval/openstax_american_yawp_qa_dataset.csv` is a faithful mapping of the upstream QA pairs.
- `data/raw/openstax_american_yawp/**/*.md` is a derived pseudo-corpus built by grouping answer snippets by source/chapter/file so the current RAG pipeline has something retrievable.
- This is suitable for pipeline smoke tests, schema validation, and baseline-vs-RAG comparisons on a public dataset.
- It is not equivalent to using the original textbook paragraphs as the retrieval corpus.

### Outputs

Running the adapter produces:

- `data/eval/openstax_american_yawp_qa_dataset.csv`
- `data/eval/openstax_american_yawp_qa_metadata.csv`
- `data/eval/openstax_american_yawp_mapping_summary.json`
- `data/raw/openstax_american_yawp/**/*.md`

### Usage

```bash
python scripts/datasets/openstax_american_yawp/prepare_dataset.py
```

To cap the number of evaluation questions:

```bash
python scripts/datasets/openstax_american_yawp/prepare_dataset.py --max-questions 300
```

Then run the project with:

```bash
python scripts/build_index.py --config configs/openstax_american_yawp_local.yaml
python scripts/run_generation.py --config configs/openstax_american_yawp_local.yaml --experiment-id <experiment_id>
python scripts/run_evaluation.py --config configs/openstax_american_yawp_local.yaml --experiment-id <experiment_id>
```
