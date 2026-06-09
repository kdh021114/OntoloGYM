from __future__ import annotations

"""
A CLI module that runs Termo to extract terms, acronyms, definitions, and relationships from a text file.
"""
import argparse
import logging
import sys
from pathlib import Path

from utils import read_text, read_tuples_list_from_csv, write_tuples_list_to_csv

sys.path.insert(0, str(Path(__file__).resolve().parent / "termo"))
from termo import Termo

logging.basicConfig(level=logging.INFO)


def _output_paths(text_file: Path, output_dir: Path) -> dict[str, Path]:
    return {
        "terms": output_dir / (text_file.stem + ".terms.csv"),
        "definitions": output_dir / (text_file.stem + ".definitions.csv"),
        "acronyms": output_dir / (text_file.stem + ".acronyms.csv"),
        "relationships": output_dir / (text_file.stem + ".relationships.csv"),
    }


def _load_terms(term_file: Path) -> list[tuple]:
    if not term_file.exists():
        return []
    return read_tuples_list_from_csv(term_file)


def _load_acronyms(acro_file: Path) -> dict[str, str]:
    if not acro_file.exists():
        return {}
    return {row[0]: row[1] for row in read_tuples_list_from_csv(acro_file) if len(row) >= 2}


def _model_options(
    temperature=None,
    num_ctx=None,
    max_completion_tokens=None,
    reasoning_effort=None,
    base_url=None,
):
    options = {}
    if temperature is not None:
        options["temperature"] = temperature
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    if max_completion_tokens is not None:
        options["max_completion_tokens"] = max_completion_tokens
    if reasoning_effort is not None:
        options["reasoning_effort"] = reasoning_effort
    if base_url is not None:
        options["base_url"] = base_url
    return options


def _new_termo(
    text,
    backend,
    context_kind,
    relationship_mode,
    include_literal_values,
    preserve_table_blocks,
    terms=None,
    acronyms=None,
):
    termo = Termo(
        text,
        backend=backend,
        context_kind=context_kind,
        relationship_mode=relationship_mode,
        include_literal_values=include_literal_values,
        preserve_table_blocks=preserve_table_blocks,
    )
    termo["terms"] = list(terms or [])
    termo["acronyms"] = dict(acronyms or {})
    return termo


def extract_termo_artifacts(
    text_file,
    output_dir=None,
    terms_model=None,
    acronyms_model=None,
    definitions_model=None,
    relationships_model=None,
    max_length_split_terms=2000,
    max_length_split_acronyms=2000,
    max_length_split_definitions=2000,
    max_length_split_relationships=2000,
    terms_model_params=None,
    acronyms_model_params=None,
    definitions_model_params=None,
    relationships_model_params=None,
    terms_backend="ollama",
    acronyms_backend=None,
    definitions_backend=None,
    relationships_backend=None,
    context_kind="generic",
    relationship_mode="generic",
    include_literal_values=False,
    preserve_table_blocks=False,
    remove_hallucinated=True,
    run_terms=True,
    run_acronyms=True,
    run_definitions=True,
    run_relationships=True,
):
    """
    Run TERMO artifact extraction as independently configurable steps.
    """
    text_file = Path(text_file)
    output_dir = Path(output_dir) if output_dir else text_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _output_paths(text_file, output_dir)

    text = read_text(text_file)
    acronyms_backend = acronyms_backend or terms_backend
    definitions_backend = definitions_backend or terms_backend
    relationships_backend = relationships_backend or terms_backend
    terms_model_params = terms_model_params or {}
    acronyms_model_params = acronyms_model_params or terms_model_params
    definitions_model_params = definitions_model_params or terms_model_params
    relationships_model_params = relationships_model_params or terms_model_params

    terms = _load_terms(paths["terms"])
    acronyms = _load_acronyms(paths["acronyms"])

    if run_terms:
        logging.info("Extracting terms from %s using %s/%s.", text_file, terms_backend, terms_model)
        termo = _new_termo(
            text,
            backend=terms_backend,
            context_kind=context_kind,
            relationship_mode=relationship_mode,
            include_literal_values=include_literal_values,
            preserve_table_blocks=preserve_table_blocks,
        )
        terms = termo.extract_terms(
            model=terms_model,
            max_length_split=max_length_split_terms,
            remove_hallucinated=remove_hallucinated,
            options=terms_model_params,
        )
        write_tuples_list_to_csv(paths["terms"], terms)
    elif not terms:
        logging.warning("Skipping term extraction but no existing terms CSV was found at %s.", paths["terms"])
        write_tuples_list_to_csv(paths["terms"], [])

    if run_acronyms:
        logging.info("Extracting acronyms from %s using %s/%s.", text_file, acronyms_backend, acronyms_model)
        termo = _new_termo(
            text,
            backend=acronyms_backend,
            context_kind=context_kind,
            relationship_mode=relationship_mode,
            include_literal_values=include_literal_values,
            preserve_table_blocks=preserve_table_blocks,
            terms=terms,
        )
        acronyms = termo.extract_acronyms(
            model=acronyms_model,
            max_length_split=max_length_split_acronyms,
            options=acronyms_model_params,
        )
        write_tuples_list_to_csv(paths["acronyms"], list(acronyms.items()))
    elif not paths["acronyms"].exists():
        logging.info("Skipping acronym extraction and writing an empty acronyms CSV at %s.", paths["acronyms"])
        write_tuples_list_to_csv(paths["acronyms"], [])

    if run_definitions:
        logging.info("Extracting definitions from %s using %s/%s.", text_file, definitions_backend, definitions_model)
        termo = _new_termo(
            text,
            backend=definitions_backend,
            context_kind=context_kind,
            relationship_mode=relationship_mode,
            include_literal_values=include_literal_values,
            preserve_table_blocks=preserve_table_blocks,
            terms=terms,
        )
        definitions = termo.extract_definitions(
            model=definitions_model,
            max_length_split=max_length_split_definitions,
            options=definitions_model_params,
        )
        write_tuples_list_to_csv(paths["definitions"], list(definitions.items()))
    elif not paths["definitions"].exists():
        logging.info("Skipping definition extraction and writing an empty definitions CSV at %s.", paths["definitions"])
        write_tuples_list_to_csv(paths["definitions"], [])

    if run_relationships:
        logging.info(
            "Extracting relationships from %s using %s/%s.",
            text_file,
            relationships_backend,
            relationships_model,
        )
        termo = _new_termo(
            text,
            backend=relationships_backend,
            context_kind=context_kind,
            relationship_mode=relationship_mode,
            include_literal_values=include_literal_values,
            preserve_table_blocks=preserve_table_blocks,
            terms=terms,
            acronyms=acronyms,
        )
        relationships = termo.extract_relationships(
            model=relationships_model,
            max_length_split=max_length_split_relationships,
            options=relationships_model_params,
        )
        write_tuples_list_to_csv(paths["relationships"], relationships)
    elif not paths["relationships"].exists():
        logging.info("Skipping relationship extraction and writing an empty relationships CSV at %s.", paths["relationships"])
        write_tuples_list_to_csv(paths["relationships"], [])

    return paths


def extract_terms(
    text_file,
    model=None,
    output_dir=None,
    max_length_split_terms=2000,
    max_length_split_acronyms=2000,
    max_length_split_definitions=2000,
    max_length_split_relationships=2000,
    model_params=None,
    backend="ollama",
    context_kind="generic",
    relationship_mode="generic",
    include_literal_values=False,
    preserve_table_blocks=False,
    remove_hallucinated=True,
    skip_acronyms=False,
    skip_definitions=False,
    skip_relationships=False,
):
    """
    Backward-compatible TERMO runner using one model/backend for every step.
    """
    return extract_termo_artifacts(
        text_file=text_file,
        output_dir=output_dir,
        terms_model=model,
        acronyms_model=model,
        definitions_model=model,
        relationships_model=model,
        max_length_split_terms=max_length_split_terms,
        max_length_split_acronyms=max_length_split_acronyms,
        max_length_split_definitions=max_length_split_definitions,
        max_length_split_relationships=max_length_split_relationships,
        terms_model_params=model_params or {},
        acronyms_model_params=model_params or {},
        definitions_model_params=model_params or {},
        relationships_model_params=model_params or {},
        terms_backend=backend,
        acronyms_backend=backend,
        definitions_backend=backend,
        relationships_backend=backend,
        context_kind=context_kind,
        relationship_mode=relationship_mode,
        include_literal_values=include_literal_values,
        preserve_table_blocks=preserve_table_blocks,
        remove_hallucinated=remove_hallucinated,
        run_terms=True,
        run_acronyms=not skip_acronyms,
        run_definitions=not skip_definitions,
        run_relationships=not skip_relationships,
    )


def run_termo_on_file(
    input_path: Path,
    output_dir: Path,
    model: str,
    backend: str = "ollama",
    context_kind: str = "generic",
    relationship_mode: str = "generic",
    include_literal_values: bool = False,
    preserve_table_blocks: bool = False,
    temperature: float = 0.1,
    num_ctx: int = 9000,
    max_length_split_terms: int = 2000,
    max_length_split_acronyms: int = 2000,
    max_length_split_definitions: int = 2000,
    max_length_split_relationships: int = 2000,
    remove_hallucinated: bool = True,
    skip_acronyms: bool = False,
    skip_definitions: bool = False,
    skip_relationships: bool = False,
) -> dict[str, Path]:
    model_params = _model_options(temperature=temperature, num_ctx=num_ctx)
    return extract_terms(
        input_path,
        model=model,
        output_dir=output_dir,
        max_length_split_terms=max_length_split_terms,
        max_length_split_acronyms=max_length_split_acronyms,
        max_length_split_definitions=max_length_split_definitions,
        max_length_split_relationships=max_length_split_relationships,
        model_params=model_params,
        backend=backend,
        context_kind=context_kind,
        relationship_mode=relationship_mode,
        include_literal_values=include_literal_values,
        preserve_table_blocks=preserve_table_blocks,
        remove_hallucinated=remove_hallucinated,
        skip_acronyms=skip_acronyms,
        skip_definitions=skip_definitions,
        skip_relationships=skip_relationships,
    )


def run_termo_steps_on_file(
    input_path: Path,
    output_dir: Path,
    terms_model: str,
    acronyms_model: str,
    definitions_model: str,
    relationships_model: str,
    terms_backend: str = "ollama",
    acronyms_backend: str | None = None,
    definitions_backend: str | None = None,
    relationships_backend: str | None = None,
    context_kind: str = "generic",
    relationship_mode: str = "generic",
    include_literal_values: bool = False,
    preserve_table_blocks: bool = False,
    terms_temperature: float | None = 0.1,
    acronyms_temperature: float | None = 0.1,
    definitions_temperature: float | None = 0.1,
    relationships_temperature: float | None = 0.1,
    terms_num_ctx: int | None = 9000,
    acronyms_num_ctx: int | None = 9000,
    definitions_num_ctx: int | None = 9000,
    relationships_num_ctx: int | None = 9000,
    terms_max_completion_tokens: int | None = None,
    acronyms_max_completion_tokens: int | None = None,
    definitions_max_completion_tokens: int | None = None,
    relationships_max_completion_tokens: int | None = None,
    terms_reasoning_effort: str | None = None,
    acronyms_reasoning_effort: str | None = None,
    definitions_reasoning_effort: str | None = None,
    relationships_reasoning_effort: str | None = None,
    terms_base_url: str | None = None,
    acronyms_base_url: str | None = None,
    definitions_base_url: str | None = None,
    relationships_base_url: str | None = None,
    max_length_split_terms: int = 2000,
    max_length_split_acronyms: int = 2000,
    max_length_split_definitions: int = 2000,
    max_length_split_relationships: int = 2000,
    remove_hallucinated: bool = True,
    run_terms: bool = True,
    run_acronyms: bool = True,
    run_definitions: bool = True,
    run_relationships: bool = True,
) -> dict[str, Path]:
    return extract_termo_artifacts(
        text_file=input_path,
        output_dir=output_dir,
        terms_model=terms_model,
        acronyms_model=acronyms_model,
        definitions_model=definitions_model,
        relationships_model=relationships_model,
        max_length_split_terms=max_length_split_terms,
        max_length_split_acronyms=max_length_split_acronyms,
        max_length_split_definitions=max_length_split_definitions,
        max_length_split_relationships=max_length_split_relationships,
        terms_model_params=_model_options(
            temperature=terms_temperature,
            num_ctx=terms_num_ctx,
            max_completion_tokens=terms_max_completion_tokens,
            reasoning_effort=terms_reasoning_effort,
            base_url=terms_base_url,
        ),
        acronyms_model_params=_model_options(
            temperature=acronyms_temperature,
            num_ctx=acronyms_num_ctx,
            max_completion_tokens=acronyms_max_completion_tokens,
            reasoning_effort=acronyms_reasoning_effort,
            base_url=acronyms_base_url,
        ),
        definitions_model_params=_model_options(
            temperature=definitions_temperature,
            num_ctx=definitions_num_ctx,
            max_completion_tokens=definitions_max_completion_tokens,
            reasoning_effort=definitions_reasoning_effort,
            base_url=definitions_base_url,
        ),
        relationships_model_params=_model_options(
            temperature=relationships_temperature,
            num_ctx=relationships_num_ctx,
            max_completion_tokens=relationships_max_completion_tokens,
            reasoning_effort=relationships_reasoning_effort,
            base_url=relationships_base_url,
        ),
        terms_backend=terms_backend,
        acronyms_backend=acronyms_backend,
        definitions_backend=definitions_backend,
        relationships_backend=relationships_backend,
        context_kind=context_kind,
        relationship_mode=relationship_mode,
        include_literal_values=include_literal_values,
        preserve_table_blocks=preserve_table_blocks,
        remove_hallucinated=remove_hallucinated,
        run_terms=run_terms,
        run_acronyms=run_acronyms,
        run_definitions=run_definitions,
        run_relationships=run_relationships,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract vocabulary, definitions, acronyms and relationships from a txt file."
    )
    parser.add_argument("model", type=str, help="Ollama model tag to use.")
    parser.add_argument(
        "txt_file", type=str, help="Path to the text file to process."
    )
    parser.add_argument(
        "--max_length_split_terms",
        help="Maximum length (in characters) of the text to split for terms extraction.",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--max_length_split_acronyms",
        help="Maximum length (in characters) of the text to split for acronyms extraction.",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--max_length_split_definitions",
        help="Maximum length (in characters) of the text to split for definitions extraction.",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--max_length_split_relationships",
        help="Maximum length (in characters) of the text to split for relationships extraction.",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--temperature", "-t", help="Model temperature to use.", type=float
    )
    parser.add_argument(
        "--num_ctx", "-n", help="Context length in tokens to use.", type=int
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "anthropic", "openai", "oai"],
        default="ollama",
        help="LLM backend to use.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory where TERMO CSV files should be written.",
    )
    parser.add_argument(
        "--context-kind",
        choices=["generic", "evidence", "table", "enriched"],
        default="generic",
        help="Context prompt mode.",
    )
    parser.add_argument(
        "--relationship-mode",
        choices=["generic", "evidence"],
        default="generic",
        help="Relationship extraction mode.",
    )
    parser.add_argument(
        "--include-literal-values",
        action="store_true",
        help="Allow explicit numeric/unit literals as relationship objects.",
    )
    parser.add_argument(
        "--preserve-table-blocks",
        action="store_true",
        help="Keep short [TABLE ...] blocks intact during chunking.",
    )
    parser.add_argument(
        "--no-remove-hallucinated",
        action="store_true",
        help="Keep LLM terms that cannot be matched back to the context.",
    )
    parser.add_argument(
        "--skip-acronyms",
        "-a",
        help="Skip acronyms extraction.",
        action="store_true",
    )
    parser.add_argument(
        "--skip-definitions",
        "-d",
        help="Skip definitions extraction.",
        action="store_true",
    )
    parser.add_argument(
        "--skip-relationships",
        "-r",
        help="Skip relationships extraction.",
        action="store_true",
    )

    args = parser.parse_args()
    model = args.model
    model_params = {}
    if args.temperature:
        model_params["temperature"] = args.temperature
    if args.num_ctx:
        model_params["num_ctx"] = args.num_ctx

    extract_terms(
        text_file=args.txt_file,
        model=model,
        output_dir=args.output_dir,
        max_length_split_terms=args.max_length_split_terms,
        max_length_split_acronyms=args.max_length_split_acronyms,
        max_length_split_definitions=args.max_length_split_definitions,
        max_length_split_relationships=args.max_length_split_relationships,
        model_params=model_params,
        backend=args.backend,
        context_kind=args.context_kind,
        relationship_mode=args.relationship_mode,
        include_literal_values=args.include_literal_values,
        preserve_table_blocks=args.preserve_table_blocks,
        remove_hallucinated=not args.no_remove_hallucinated,
        skip_acronyms=args.skip_acronyms,
        skip_definitions=args.skip_definitions,
        skip_relationships=args.skip_relationships,
    )
