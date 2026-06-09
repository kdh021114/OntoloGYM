# OntoloGYM

This folder keeps the extraction workflows separated while letting them share
paper PDFs and one environment file.

- `qa_extractor/`: copied AirQA extractor code plus the support modules it uses.
- `ontogen/`: copied OntoGen code, excluding generated examples, notebooks,
  caches, and bundled sample data.
- `relation_augmentation/`: evidence-based KG relation augmentation pipeline.
- `qa_evaluation/`: AirQA questions answered with KG context and scored with AirQA evaluators.
- `configs/`: user-editable settings split by pipeline.
- `data/papers/`: shared input folder. Folder-style inputs are preferred:
  `Journal_or_Conference_0/<paper json>` plus `figures/`.
- `data/<run_id>/`: connected outputs from one experiment run.
- `config.py`: compatibility layer that re-exports `configs/`.
- `.env`: shared API/provider secrets, based on `.env.example`.

The paper data formats are intentionally not converted here. Parsed JSON is
normalized only inside each run folder. Image paths are preserved in
`input_papers.json`, AirQA metadata, and OntoGen `processed_data` for later
multimodal work.

Run from this folder:

```bash
python run_pipeline_sequence.py
python run_qa_extractor.py
python run_ontogen.py
python run_relation_augmentation.py
python run_qa_evaluation.py
python run_kg_visualization.py
```

All runners read `configs/` through `config.py`; no command-line arguments are
required. `run_pipeline_sequence.py` creates a new `data/<run_id>/` folder
and runs the configured order. Individual runners reuse the active run recorded
in `data/_active_run.txt`, unless `configs/common.py` is set to create a
new run.

AirQA question counts are controlled in `configs/qa_extractor.py` with
`QA_TYPE_EXAMPLE_COUNTS`. The supported keys are `single`, `multi`, `rag`
(AirQA retrieval questions), and `comprehensive`. Set a type's count to `0` to
skip it. `multi` questions combine two available `single` questions, so generate
enough `single` examples first or keep existing single examples in
`data/airqa/examples`.

Generated AirQA examples also include short verification fields: `answer`,
`context`, and `reasoning`. These are intentionally concise and are not used by
the official AirQA evaluator.

OntoGen category generation includes an optional LLM curation step controlled in
`configs/ontogen.py`. By default it follows the paper's manual pruning ratio
`12/131`, so many candidate categories are compressed into a smaller seed list
before taxonomy generation.
