# Evidence-Aware OntoGen Pipeline

This extension broadens the original abstract/introduction workflow into an
evidence-aware and table-aware ontology/KG generation pipeline.

## Motivation

The original OntoGen path focuses mainly on abstract and introduction text. That
is useful for high-level terms, but many ontology and KG signals appear in
methods, experiment descriptions, results, ablation studies, and PDF tables.
This pipeline keeps section and table provenance markers so downstream TERMO
prompts can extract relationships grounded in explicit evidence.

This extension does not include evaluation, QA benchmark creation, LLM judging,
figure interpretation, chart understanding, or multimodal vision model logic.

## Primary Usage

Use the config-driven workflow:

1. Put PDFs in `data/pdfs/`.
2. Edit `config.py`, especially `PDF_FILES`, `DOMAIN_NAME`,
   `TEXT_SOURCE`, `SECTIONS_TO_EXTRACT`, and the `RUN_*` flags.
3. Run:

```bash
python run.py
```

The default `config.py` keeps expensive steps disabled. Turn on the steps you
want to run:

```python
RUN_PDF_TO_TEXT = True
RUN_SECTION_EXTRACTION = True
RUN_TABLE_EXTRACTION = True
RUN_BUILD_ENRICHED_CONTEXT = True
RUN_TERMO = True
```

For category and taxonomy generation, also enable:

```python
RUN_CATEGORY_GENERATION = True
RUN_TAXONOMY_GENERATION = True
```

## Key Config Settings

`PDF_FILES`
: PDF names or paths. If empty, `run.py` processes every `*.pdf` in `PDF_DIR`.

`TEXT_SOURCE`
: `"nougat"` or `"pymupdf"`.

`SECTIONS_TO_EXTRACT`
: Supported sections include `abstract`, `introduction`, `methods`,
  `experimental`, `experiments`, `evaluation`, `results`, `discussion`, and
  `ablation`.

`EXTRACT_TABLES`, `INCLUDE_TABLES_IN_ENRICHED_CONTEXT`
: Control PDF table extraction and whether serialized tables are appended to the
  enriched context.

`CONTEXT_KIND`, `RELATIONSHIP_MODE`
: Use `CONTEXT_KIND = "enriched"` and `RELATIONSHIP_MODE = "evidence"` for the
  evidence-aware TERMO prompts.

`INCLUDE_LITERAL_VALUES`
: Allows explicit numeric/unit values as relationship objects when they appear
  in the context.

## AirQA-Style Processed Data

This pipeline now includes a lightweight `processed_data` intermediate format
inspired by AirQA-style parser outputs. It stores parser-agnostic document
structure in `data/processed_data/<paper_id>.processed_data.json`, including
sections, tables, optional figure captions, optional equations, and metadata.

OntoGen does not run MinerU or Docling directly. If an external AirQA/MinerU-style
JSON already exists, set `PROCESSED_DATA_SOURCE = "airqa_mineru"` or `"auto"` and
point `AIRQA_PROCESSED_DATA_DIR` to the folder containing that JSON. Supported
inputs include `info_from_mineru.TOC`, `info_from_mineru.tables`,
`info_from_mineru.figures`, `info_from_mineru.equations`, and equivalent
top-level `TOC`, `tables`, `figures`, and `equations` keys.

If no external parser JSON is available, keep `PROCESSED_DATA_SOURCE = "existing"`.
The existing PyMuPDF/Nougat section extraction and table extraction outputs are
converted into the same `processed_data` schema, then enriched context can be
built from that normalized JSON.

`USE_PROCESSED_DATA_FOR_ENRICHED_CONTEXT = True` makes `run.py` prefer
processed_data when building enriched context. If processed_data is missing or
cannot be built, the pipeline falls back to the original section JSONL + table
JSONL enriched-context path.

Figure handling is limited to captions and provenance. Image understanding,
image crops, visual QA, plot reading, and chart value extraction are not
implemented.

## Outputs

`data/text/`
: Processed text files such as `<paper>.processed.pymupdf.txt`.

`data/sections/`
: Section text files and `<paper>.processed.<source>.sections.jsonl`. JSONL
  records include `source_file`, `paper_id`, `section`, `heading`, `content`,
  `start_line`, and `end_line`.

`data/tables/`
: `<paper>.tables.jsonl` and `<paper>.tables.md`. Table records include
  `source_file`, `paper_id`, `page`, `table_index`, `caption`, `columns`, and
  `rows`.

`data/processed_data/`
: Normalized parser-agnostic JSON files such as
  `<paper>.processed_data.json`.

`data/enriched/`
: One context file per paper with provenance markers such as
  `[PAPER=...][SECTION=results]` and `[TABLE paper_id=... page=... table=...]`.

`data/termo/`
: TERMO CSV files following the existing naming convention:
  `<input_stem>.terms.csv`, `<input_stem>.acronyms.csv`,
  `<input_stem>.definitions.csv`, and `<input_stem>.relationships.csv`.

`data/categories/`
: Category seed files.

`data/taxonomy/`
: Taxonomy pickle files.

## Advanced CLI Usage

The existing CLI scripts are still available for backward compatibility:

```bash
python extract_sections.py --nougat --abstract --introduction docs/paper.processed.nougat.txt
python run_termo.py llama3.1:70b docs/paper.processed.nougat.abstract.txt
```

New evidence-aware options can also be used directly:

```bash
python extract_sections.py --pymupdf --sections abstract methods results ablation docs/paper.processed.pymupdf.txt
python extract_tables.py docs/paper.pdf --output-dir data/tables
python build_enriched_context.py data/sections/paper.processed.pymupdf.sections.jsonl data/enriched/paper.enriched.txt --tables-jsonl data/tables/paper.tables.jsonl
python run_termo.py llama3.1:70b data/enriched/paper.enriched.txt --context-kind enriched --relationship-mode evidence --include-literal-values --preserve-table-blocks
```

## Known Limitations

Table extraction quality depends on PDF layout and the installed PyMuPDF
version.

Table captions are extracted only when an explicit nearby `Table N...` caption is
detected.

Figure, image, and chart understanding are not implemented.

LLM output still needs human curation.

Evidence KG relations are different from `isA` taxonomy relations.

Numeric values should be treated as evidence relationship objects, not ontology
classes.
