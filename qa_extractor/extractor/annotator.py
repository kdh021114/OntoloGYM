#coding=utf
import ast
import os, sys, logging, json, random, re, uuid, html
from datetime import datetime
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Type, Union, Optional
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
ONTOLOGYM_ROOT = Path(__file__).resolve().parents[2]
if os.fspath(ONTOLOGYM_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(ONTOLOGYM_ROOT))
from runtime import load_config
from common.papers import PaperInput, discover_paper_inputs, write_paper_manifest
from common.run_context import record_pipeline_run
import extractor.annotator
import extractor.explorer, extractor.tracker
from extractor.explorer import BaseExplorer, SingleExplorer, MultipleExplorer, RetrievalExplorer, ComprehensiveExplorer
from extractor.tracker import BaseTracker, SingleTracker, MultipleTracker, RetrievalTracker, ComprehensiveTracker
from evaluation.evaluator import evaluate_airqa
from utils.llm_utils import call_llm, DEFAULT_LLM_MODEL, DEFAULT_TOP_P, DEFAULT_TEMPERATURE

qa_config = load_config()
qa_config.ensure_directories()

DATA_DIR = os.fspath(qa_config.DATA_DIR)
EXAMPLE_DIR = os.fspath(qa_config.EXAMPLE_DIR)
METADATA_DIR = os.fspath(qa_config.METADATA_DIR)
PAPER_DIR = os.fspath(qa_config.PAPER_DIR)
PROCESSED_DIR = os.fspath(qa_config.PROCESSED_DATA_DIR)
EXPLORE_UUID_ATTEMP = 5
EXPLORE_NO_UUID_ATTEMP = 10

def get_all_example_uuids(example_dir: str = EXAMPLE_DIR) -> List[str]:
    """ Extract all example UUIDs.
    """
    if not os.path.isdir(example_dir):
        return []
    return [os.path.splitext(f)[0] for f in os.listdir(example_dir) if f.endswith('.json')]

def get_uuid(name: Optional[str] = None, uuid_type: str = 'uuid5', uuid_namespace: str = 'dns') -> str:
    """ Generate a UUID string given the input name.
    """
    namespaces = {'dns': uuid.NAMESPACE_DNS, 'url': uuid.NAMESPACE_URL, 'oid': uuid.NAMESPACE_OID, 'x500': uuid.NAMESPACE_X500}
    namespace = namespaces[uuid_namespace.lower()]
    if uuid_type == 'uuid3' or uuid_type == 'uuid5':
        if name is None:
            raise ValueError('The input name should not be None for uuid3 or uuid5.')

        uid = uuid.uuid5(namespace, name) if uuid_type == 'uuid5' else uuid.uuid3(namespace, name)
    else:
        uid = uuid.uuid4()
    return str(uid)

def generate_airqa_example_template(example_dir: str = EXAMPLE_DIR, **kwargs) -> Dict[str, Any]:
    """ Generate an AirQA example template.
    """
    os.makedirs(example_dir, exist_ok=True)
    flag, existing_uids = True, get_all_example_uuids(example_dir=example_dir)
    while flag:
        uid = get_uuid(name=os.path.abspath(__file__) + str(os.urandom(8)))
        if uid not in existing_uids:
            break
    example_template = {
        "uuid": uid,
        "question": "",
        "answer_format": "Your answer should be ",
        "tags": [],
        "anchor_pdf": [],
        "reference_pdf": [],
        "conference": [],
        "evaluator": {
            "eval_func": "eval_",
            "eval_kwargs": {}
        },
        "annotator": "human"
    }
    example_template.update(kwargs)
    with open(os.path.join(example_dir, uid + '.json'), 'w', encoding='utf-8') as ouf:
        json.dump(example_template, ouf, ensure_ascii=False, indent=4)
    print(f"Generated an AIR-QA example template with ID {uid} into {example_dir}/{uid}.json file.")
    return example_template


TRACKER_FIELD_PATTERN = re.compile(
    r"\[(question|evaluator|answer_format|answer|tag)\]\s*:\s*",
    re.IGNORECASE,
)


def parse_tracker_response(content: str) -> Optional[Dict[str, str]]:
    """Parse tracker fields even when nano swaps answer_format/evaluator order."""
    fenced_blocks = [
        match.group(1)
        for match in re.finditer(r"```(?:txt|json)?\s*(.*?)```", content, re.DOTALL | re.IGNORECASE)
    ]
    candidates = fenced_blocks + [content]
    required = {"question", "evaluator", "answer_format", "answer", "tag"}
    for text in candidates:
        parsed = _parse_tracker_fields(text)
        if required.issubset(parsed):
            return parsed
    return None


def _parse_tracker_fields(text: str) -> Dict[str, str]:
    matches = list(TRACKER_FIELD_PATTERN.finditer(text))
    if not matches:
        return {}
    fields: Dict[str, str] = {}
    for index, match in enumerate(matches):
        key = match.group(1).lower()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        fields[key] = text[start:end].strip()
    return fields


def parse_evaluator_field(
    raw_evaluator: str,
    question: str = "",
    answer: Any = "",
) -> Optional[Dict[str, Any]]:
    try:
        evaluator = json.loads(raw_evaluator)
    except json.JSONDecodeError:
        try:
            evaluator = ast.literal_eval(raw_evaluator)
        except (SyntaxError, ValueError):
            repaired = re.sub(r",\s*([}\]])", r"\1", raw_evaluator)
            try:
                evaluator = json.loads(repaired)
            except json.JSONDecodeError:
                function_name = raw_evaluator.strip().strip("`").strip()
                if re.fullmatch(r"eval_[A-Za-z0-9_]+", function_name):
                    return default_evaluator_for_function_name(function_name, question, answer)
                return None
    if isinstance(evaluator, dict):
        return evaluator
    if isinstance(evaluator, str) and re.fullmatch(r"eval_[A-Za-z0-9_]+", evaluator.strip()):
        return default_evaluator_for_function_name(evaluator.strip(), question, answer)
    return None


def default_evaluator_for_function_name(function_name: str, question: str, answer: Any) -> Optional[Dict[str, Any]]:
    if function_name == "eval_reference_answer_with_llm":
        return {
            "eval_func": function_name,
            "eval_kwargs": {
                "reference_answer": str(answer),
                "question": question,
            },
        }
    if function_name == "eval_string_exact_match":
        return {
            "eval_func": function_name,
            "eval_kwargs": {
                "gold": str(answer),
                "lowercase": True,
                "ignore_blank": True,
            },
        }
    if function_name == "eval_int_exact_match":
        try:
            gold = int(float(str(answer).strip()))
        except ValueError:
            return None
        return {"eval_func": function_name, "eval_kwargs": {"gold": gold}}
    if function_name == "eval_float_exact_match":
        try:
            gold = float(str(answer).strip())
        except ValueError:
            return None
        return {"eval_func": function_name, "eval_kwargs": {"gold": gold}}
    if function_name == "eval_bool_exact_match":
        lowered = str(answer).strip().lower()
        if lowered not in {"true", "false"}:
            return None
        return {"eval_func": function_name, "eval_kwargs": {"gold": lowered == "true"}}
    return None


def coerce_answer_for_evaluator(answer: Any, evaluator: Dict[str, Any]) -> Any:
    eval_func = evaluator.get("eval_func")
    kwargs = evaluator.get("eval_kwargs", {})
    if eval_func in {
        "eval_string_exact_match",
        "eval_int_exact_match",
        "eval_float_exact_match",
        "eval_bool_exact_match",
    } and "gold" in kwargs:
        return kwargs["gold"]
    if eval_func == "eval_structured_object_exact_match" and "gold" in kwargs:
        return json.dumps(kwargs["gold"], ensure_ascii=False)
    return answer

def get_qids(tags: str):
    uuids = []
    if not os.path.isdir(EXAMPLE_DIR):
        return uuids
    for example in os.listdir(EXAMPLE_DIR):
        if example.endswith(".json"):
            example = json.load(open(os.path.join(EXAMPLE_DIR, example), "r", encoding="utf-8"))
            if tags in example["tags"]:
                uuids.append(example["uuid"])
    return uuids

def get_pids():
    uuids = []
    if not os.path.isdir(METADATA_DIR):
        return uuids
    for metadata in os.listdir(METADATA_DIR):
        if metadata.endswith(".json"):
            uuids.append(metadata.split(".")[0])
    return uuids

def get_used_qids():
    uuids = []
    if not os.path.isdir(EXAMPLE_DIR):
        return uuids
    for example in os.listdir(EXAMPLE_DIR):
        if example.endswith(".json"):
            example = json.load(open(os.path.join(EXAMPLE_DIR, example), "r", encoding="utf-8"))
            if "from" in example:
                for qid in example["from"]:
                    uuids.append(qid)
    return uuids

def get_used_pids():
    uuids = []
    if not os.path.isdir(EXAMPLE_DIR):
        return uuids
    for example in os.listdir(EXAMPLE_DIR):
        if example.endswith(".json"):
            example = json.load(open(os.path.join(EXAMPLE_DIR, example), "r", encoding="utf-8"))
            for pid in example["anchor_pdf"]:
                uuids.append(pid)
            for pid in example["reference_pdf"]:
                uuids.append(pid)
    return uuids


def _clean_metadata_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _section_by_title(toc: List[Dict[str, Any]], title_keyword: str) -> str:
    for section in toc:
        title = _clean_metadata_text(section.get("title", "")).lower()
        if title_keyword.lower() in title:
            return _clean_metadata_text(section.get("text", ""))
    return ""


def _title_from_parsed_json(raw_data: Dict[str, Any], fallback: str) -> str:
    metadata = raw_data.get("metadata") if isinstance(raw_data.get("metadata"), dict) else {}
    title = _clean_metadata_text(metadata.get("title", ""))
    if title:
        return title

    toc = raw_data.get("TOC")
    if not isinstance(toc, list):
        toc = raw_data.get("info_from_mineru", {}).get("TOC", [])
    if not isinstance(toc, list):
        toc = raw_data.get("sections", [])
    for section in toc if isinstance(toc, list) else []:
        candidate = _clean_metadata_text(section.get("title", ""))
        if candidate and "abstract" not in candidate.lower() and len(candidate) > 20:
            return candidate
    return fallback


def _metadata_from_parsed_json(raw_data: Dict[str, Any], paper_input: PaperInput) -> Dict[str, Any]:
    metadata = raw_data.get("metadata") if isinstance(raw_data.get("metadata"), dict) else {}
    toc = raw_data.get("TOC")
    if not isinstance(toc, list):
        toc = raw_data.get("info_from_mineru", {}).get("TOC", [])
    if not isinstance(toc, list):
        toc = raw_data.get("sections", [])
    if not isinstance(toc, list):
        toc = []

    conference = _clean_metadata_text(metadata.get("conference", ""))
    if not conference:
        conference = _clean_metadata_text(paper_input.collection)

    return {
        "title": _title_from_parsed_json(raw_data, paper_input.paper_id),
        "abstract": _clean_metadata_text(metadata.get("abstract", "")) or _section_by_title(toc, "abstract"),
        "conference": conference,
        "year": metadata.get("year", ""),
        "paper_assets": paper_input.asset_payload(),
    }


def prepare_parsed_json_inputs() -> int:
    """Copy parsed paper JSON files into the AirQA metadata/processed-data layout."""
    paper_root = Path(PAPER_DIR)
    if not paper_root.exists():
        return 0

    prepared = 0
    Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
    Path(METADATA_DIR).mkdir(parents=True, exist_ok=True)

    paper_inputs = discover_paper_inputs(paper_root)
    if getattr(qa_config, "RUN_OUTPUT_DIR", None):
        write_paper_manifest(paper_inputs, Path(qa_config.RUN_OUTPUT_DIR) / "input_papers.json")

    for paper_input in paper_inputs:
        source_path = paper_input.json_path
        with source_path.open("r", encoding="utf-8") as handle:
            raw_data = json.load(handle)
        if not isinstance(raw_data, dict):
            continue
        reused_processed_data = raw_data.get("reused_processed_data")
        if reused_processed_data:
            reused_path = Path(str(reused_processed_data))
            if reused_path.exists():
                with reused_path.open("r", encoding="utf-8") as handle:
                    raw_data = json.load(handle)
                raw_data.setdefault("reused_processed_data", os.fspath(reused_path))

        paper_id = _clean_metadata_text(raw_data.get("paper_id", "")) or paper_input.paper_id
        processed_payload = dict(raw_data)
        processed_payload.setdefault("paper_id", paper_id)
        processed_payload.setdefault("source_file", os.fspath(source_path))
        processed_payload.setdefault("paper_dir", os.fspath(paper_input.paper_dir))
        processed_payload.setdefault("pdf_path", raw_data.get("pdf_path", ""))
        processed_payload.setdefault("paper_assets", paper_input.asset_payload())

        with open(os.path.join(PROCESSED_DIR, f"{paper_id}.json"), "w", encoding="utf-8") as output:
            json.dump(processed_payload, output, ensure_ascii=False, indent=2)

        with open(os.path.join(METADATA_DIR, f"{paper_id}.json"), "w", encoding="utf-8") as output:
            json.dump(_metadata_from_parsed_json(raw_data, paper_input), output, ensure_ascii=False, indent=2)

        prepared += 1
    return prepared


def reload_available_uuids():
    global available_pids, single_qids, multiple_qids
    available_pids = get_pids()
    single_qids, multiple_qids = get_qids("single"), get_qids("multiple")


available_pids = get_pids()
single_qids, multiple_qids = get_qids("single"), get_qids("multiple")

def reset_available_uuids():
    global available_pids, single_qids, multiple_qids
    used_pids = get_used_pids()
    available_pids = [uuid for uuid in available_pids if uuid not in used_pids]

    used_qids = get_used_qids()
    single_qids, multiple_qids = get_qids("single"), get_qids("multiple")
    single_qids = [qid for qid in single_qids if qid not in used_qids]
    multiple_qids = [qid for qid in multiple_qids if qid not in used_qids]

class ParseError(Exception):
    pass

COMMON_CONTEXT_WORDS = {
    "about", "according", "answer", "between", "does", "from", "give", "have",
    "into", "paper", "provide", "question", "report", "reported", "should",
    "show", "shows", "table", "tell", "that", "their", "there", "this",
    "what", "when", "where", "which", "with", "within", "your",
}

QUALITY_STOPWORDS = COMMON_CONTEXT_WORDS | {
    "single", "sentence", "string", "number", "value", "format", "exactly",
    "return", "including", "using", "used", "section", "content", "provided",
    "paper", "answer", "format", "with", "without", "must", "only",
}

NUMBER_PATTERN = re.compile(
    r"[-+]?\d+(?:\.\d+)?(?:\s*[–-]\s*[-+]?\d+(?:\.\d+)?)?(?:\s*(?:%|K|mT|T|Pa|kPa|mW|W|h|hours?|days?|nm|µm|μm|μB|µB|meV|eV|Hz|kHz|MHz|°C|C|g|mg|mm|Å|kOe|Oe))?",
    re.IGNORECASE,
)


def _clean_text(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _answer_scalars(value: Any) -> List[str]:
    if isinstance(value, dict):
        scalars: List[str] = []
        for item in value.values():
            scalars.extend(_answer_scalars(item))
        return scalars
    if isinstance(value, (list, tuple, set)):
        scalars = []
        for item in value:
            scalars.extend(_answer_scalars(item))
        return scalars
    text = _clean_text(value)
    if not text:
        return []
    if not isinstance(value, str):
        return [text]
    try:
        parsed = json.loads(text)
    except Exception:
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = None
    if parsed is not None and parsed is not value:
        return _answer_scalars(parsed)
    return [text]


def _quality_tokens(value: Any) -> set[str]:
    tokens = set()
    for scalar in _answer_scalars(value):
        for token in re.findall(r"[A-Za-z0-9_.+%µμ°×^-]+", scalar.lower()):
            if len(token) >= 3 and token not in QUALITY_STOPWORDS:
                tokens.add(token)
    return tokens


def _quality_numbers(value: Any) -> set[str]:
    numbers = set()
    for scalar in _answer_scalars(value):
        for match in NUMBER_PATTERN.finditer(scalar):
            raw = match.group(0)
            normalized = re.sub(r"\s+", " ", raw).strip().lower()
            if normalized:
                before = scalar[match.start() - 1] if match.start() > 0 else ""
                after = scalar[match.end()] if match.end() < len(scalar) else ""
                after_next = scalar[match.end() + 1] if match.end() + 1 < len(scalar) else ""
                before_prev = scalar[match.start() - 2] if match.start() > 1 else ""
                bare_number = re.fullmatch(r"[-+]?\d+(?:\.\d+)?", normalized) is not None
                small_integer_range = re.fullmatch(
                    r"[-+]?\d+\s*[–-]\s*[-+]?\d+",
                    normalized,
                ) is not None
                formula_boundary = (
                    before.isalpha()
                    or after.isalpha()
                    or (after in {"+", "-"} and after_next.isalpha())
                    or (before in {"+", "-"} and before_prev.isalpha())
                )
                if formula_boundary:
                    continue
                if bare_number and (before.isalpha() or after.isalpha() or after in {"+", "-"}):
                    continue
                if bare_number and "." not in normalized and abs(int(float(normalized))) < 10:
                    continue
                if small_integer_range:
                    endpoints = [int(part) for part in re.findall(r"[-+]?\d+", normalized)]
                    if endpoints and all(abs(part) < 10 for part in endpoints):
                        continue
                numbers.add(normalized)
                compact = re.sub(r"\s+", "", normalized)
                if compact:
                    numbers.add(compact)
    return numbers


def _text_contains_number(text: str, number: str) -> bool:
    haystack = _clean_text(text).lower()
    compact_haystack = re.sub(r"\s+", "", haystack)
    return number in haystack or re.sub(r"\s+", "", number) in compact_haystack


def _leakage_ratio(answer: Any, visible_text: str) -> tuple[float, float]:
    answer_tokens = _quality_tokens(answer)
    visible_tokens = _quality_tokens(visible_text)
    token_ratio = len(answer_tokens & visible_tokens) / max(1, len(answer_tokens))

    answer_numbers = _quality_numbers(answer)
    leaked_numbers = {
        number for number in answer_numbers if _text_contains_number(visible_text, number)
    }
    number_ratio = len(leaked_numbers) / max(1, len(answer_numbers)) if answer_numbers else 0.0
    return token_ratio, number_ratio


def _looks_like_negative_evidence(question: str, answer: Any) -> bool:
    answer_text = _clean_text(answer).lower()
    question_text = _clean_text(question).lower()
    if "no usable content" in answer_text or "no usable content" in question_text:
        return True
    if answer_text in {"false", "no", "none", "not reported", "no."}:
        return True
    if answer_text.startswith("none") or "no reported" in answer_text or "contains no" in answer_text:
        return True
    negative_question = (
        "does the provided markdown section" in question_text
        or "are there any numerical" in question_text
        or "any quantitative" in question_text
        or "specify any experimental setup" in question_text
    )
    return negative_question and any(marker in answer_text for marker in ("false", "no", "none"))


def _question_fingerprint(question: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z0-9_.+%µμ°×^-]+", question.lower())
        if len(token) >= 3 and token not in QUALITY_STOPWORDS
    }


def _existing_question_fingerprints() -> List[set[str]]:
    fingerprints = []
    reference_dirs = [EXAMPLE_DIR]
    reference_dirs.extend(os.fspath(path) for path in getattr(qa_config, "NEAR_DUPLICATE_REFERENCE_DIRS", []))
    for example_path in reference_dirs:
        if os.path.isfile(example_path):
            try:
                with open(example_path, "r", encoding="utf-8") as handle:
                    if example_path.endswith(".jsonl"):
                        examples = [json.loads(line) for line in handle if line.strip()]
                    else:
                        payload = json.load(handle)
                        examples = payload if isinstance(payload, list) else [payload]
            except Exception:
                continue
            for example in examples:
                question = example.get("question", "") if isinstance(example, dict) else ""
                if question:
                    fingerprints.append(_question_fingerprint(question))
            continue
        if not os.path.isdir(example_path):
            continue
        for file_name in os.listdir(example_path):
            if not file_name.endswith(".json"):
                continue
            try:
                with open(os.path.join(example_path, file_name), "r", encoding="utf-8") as handle:
                    example = json.load(handle)
            except Exception:
                continue
            question = example.get("question", "")
            if question:
                fingerprints.append(_question_fingerprint(question))
    return fingerprints


def _is_near_duplicate_question(question: str) -> bool:
    current = _question_fingerprint(question)
    if not current:
        return False
    threshold = float(getattr(qa_config, "NEAR_DUPLICATE_TOKEN_OVERLAP_THRESHOLD", 0.64))
    for existing in _existing_question_fingerprints():
        if not existing:
            continue
        overlap = len(current & existing) / max(1, len(current | existing))
        if overlap >= threshold:
            return True
    return False


def validate_example_quality(question: str, answer_format: str, answer: Any) -> tuple[bool, str]:
    if not getattr(qa_config, "ENABLE_QUALITY_FILTER", True):
        return True, ""
    answer_text = _clean_text(answer)
    visible_answer_text = f"{question}\n{answer_format}\n{answer_text}".lower()
    if not answer_text or answer_text.upper() == "REJECT" or "no usable content" in visible_answer_text:
        return False, "tracker rejected the example"

    max_question_chars = int(getattr(qa_config, "MAX_QUESTION_CHARS", 0))
    max_answer_format_chars = int(getattr(qa_config, "MAX_ANSWER_FORMAT_CHARS", 0))
    max_answer_chars = int(getattr(qa_config, "MAX_ANSWER_CHARS", 0))
    if max_question_chars and len(_clean_text(question)) > max_question_chars:
        return False, f"question too long ({len(_clean_text(question))} chars)"
    if max_answer_format_chars and len(_clean_text(answer_format)) > max_answer_format_chars:
        return False, f"answer_format too long ({len(_clean_text(answer_format))} chars)"
    if max_answer_chars and len(answer_text) > max_answer_chars:
        return False, f"answer too long ({len(answer_text)} chars)"

    if getattr(qa_config, "REJECT_NEGATIVE_EVIDENCE_QUESTIONS", True) and _looks_like_negative_evidence(question, answer):
        return False, "negative-evidence/no-content question"

    visible_text = f"{question}\n{answer_format}"
    token_ratio, number_ratio = _leakage_ratio(answer, visible_text)
    token_threshold = float(getattr(qa_config, "ANSWER_LEAKAGE_TOKEN_THRESHOLD", 0.72))
    number_threshold = float(getattr(qa_config, "ANSWER_LEAKAGE_NUMERIC_THRESHOLD", 0.80))
    if len(_quality_tokens(answer)) >= 3 and token_ratio >= token_threshold:
        return False, f"answer token leakage {token_ratio:.2f}"
    answer_numbers = _quality_numbers(answer)
    answer_is_numeric = re.fullmatch(r"\s*[-+]?\d+(?:\.\d+)?\s*", answer_text) is not None
    if answer_numbers and number_ratio >= number_threshold:
        if answer_is_numeric or len(answer_numbers) >= 2 or len(answer_text) <= 60:
            return False, f"answer numeric leakage {number_ratio:.2f}"
    if _is_near_duplicate_question(question):
        return False, "near-duplicate question"
    return True, ""


def _truncate_text(text: str, max_chars: int) -> str:
    text = _clean_text(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _tokens_for_excerpt(*values: Any) -> set[str]:
    tokens = set()
    for value in values:
        for token in re.findall(r"[A-Za-z0-9_.+%-]+", str(value or "").lower()):
            if len(token) >= 3 and token not in COMMON_CONTEXT_WORDS:
                tokens.add(token)
    return tokens


def _context_units(context: str, max_unit_chars: int) -> List[str]:
    text = _clean_text(context)
    if not text:
        return []

    units = [
        unit.strip()
        for unit in re.split(r"(?<=[.!?])\s+|\s{2,}|(?:\n|\r)+", text)
        if unit.strip()
    ]
    if not units:
        units = [text]

    chunked = []
    for unit in units:
        if len(unit) <= max_unit_chars:
            chunked.append(unit)
            continue
        for start in range(0, len(unit), max_unit_chars):
            chunk = unit[start : start + max_unit_chars].strip()
            if chunk:
                chunked.append(chunk)
    return chunked


def make_concise_context(context: str, question: str, answer: Any) -> str:
    max_chars = int(getattr(qa_config, "CONTEXT_MAX_CHARS", 900))
    text = _clean_text(context)
    if not text or len(text) <= max_chars:
        return text

    keywords = _tokens_for_excerpt(question, answer)
    units = _context_units(text, max(200, max_chars // 2))
    scored = []
    for index, unit in enumerate(units):
        unit_tokens = _tokens_for_excerpt(unit)
        overlap = len(keywords & unit_tokens)
        answer_overlap = len(_tokens_for_excerpt(answer) & unit_tokens)
        score = overlap + (answer_overlap * 4)
        if score > 0:
            scored.append((score, index, unit))

    if not scored:
        return _truncate_text(text, max_chars)

    selected = []
    seen_units = set()
    used = 0
    for _, index, unit in sorted(scored, key=lambda item: (-item[0], item[1])):
        normalized_unit = unit.lower()
        if index in [existing_index for existing_index, _ in selected] or normalized_unit in seen_units:
            continue
        separator_chars = 5 if selected else 0
        if used + len(unit) + separator_chars > max_chars and selected:
            continue
        selected.append((index, unit))
        seen_units.add(normalized_unit)
        used += len(unit) + separator_chars
        if used >= max_chars:
            break

    selected_text = "\n...\n".join(unit for _, unit in sorted(selected, key=lambda item: item[0]))
    return _truncate_text(selected_text, max_chars)


def make_concise_reasoning(reasoning_steps: List[str], answer: Any) -> str:
    max_steps = int(getattr(qa_config, "REASONING_MAX_STEPS", 3))
    max_chars = int(getattr(qa_config, "REASONING_MAX_CHARS", 500))
    cleaned_steps = []
    for step in reasoning_steps:
        step = re.sub(r"^\s*[-*\d.)]+\s*", "", _clean_text(step))
        if step:
            cleaned_steps.append(step)
        if len(cleaned_steps) >= max_steps:
            break

    if cleaned_steps:
        reasoning = " ".join(f"{idx + 1}. {step}" for idx, step in enumerate(cleaned_steps))
    else:
        reasoning = f"The answer is supported by the cited context and should be reported as: {answer}"
    return _truncate_text(reasoning, max_chars)


def _format_multi_field(examples: List[Dict[str, Any]], titles: List[str], field: str, max_chars: int) -> str:
    parts = []
    per_item_chars = max(120, max_chars // max(1, len(examples)))
    for index, example in enumerate(examples):
        label = f"Paper {index + 1}"
        if index < len(titles) and titles[index]:
            label += f" ({titles[index]})"
        value = _truncate_text(example.get(field, ""), per_item_chars)
        if value:
            parts.append(f"{label}: {value}")
    return _truncate_text("\n".join(parts), max_chars)

class BaseAnnotator(ABC):
    pid: Union[str, List[str]] = None
    model: str = None
    temperature: float = None
    explorer_cls: Type[BaseExplorer] = None
    tracker_cls: Type[BaseTracker] = None

    def __init__(
            self, 
            model: str, 
            temperature: float,
            explorer_cls: Type[BaseExplorer],
            tracker_cls: Type[BaseTracker],
            **kwargs
        ):
        self.pid = kwargs.get("pid", None)
        self.model = model
        self.temperature = temperature
        self.explorer_cls = explorer_cls
        self.tracker_cls = tracker_cls
    
    @abstractmethod
    def _annotate(self, **kwargs):
        pass
    
    def annotate(self, **kwargs):
        reset_available_uuids()
        return self._annotate(**kwargs)

class SingleAnnotator(BaseAnnotator):
    pid: str = None
    def __init__(
            self, 
            model: str, 
            temperature: float, 
            explorer_cls: Type[BaseExplorer] = SingleExplorer,
            tracker_cls: Type[BaseTracker] = SingleTracker,
            **kwargs
        ):
        super().__init__(model, temperature, explorer_cls, tracker_cls, **kwargs)
    
    def _annotate(self, **kwargs):
        attemps = EXPLORE_UUID_ATTEMP if self.pid else EXPLORE_NO_UUID_ATTEMP
        for _ in range(attemps):
            try:
                if not self.pid:
                    pid = random.choice(available_pids)
                else:
                    pid = self.pid
                explorer: SingleExplorer = self.explorer_cls(pid=pid, model=self.model, temperature=self.temperature)
                explore_result = explorer.explore(**kwargs)
                messages, context, tags = explore_result[:3]
                explorer_metadata = explore_result[3] if len(explore_result) >= 4 and isinstance(explore_result[3], dict) else {}
                pattern = r"```(?:txt)?\s*.*?\[Question\]:\s*(.*?)\s*\[Answer\]:\s*(.*?)\s*\[Reasoning Steps\]:\s*(.*?)```"
                matched = re.findall(pattern, messages[-1]["content"], re.DOTALL)
                if len(matched) == 0:
                    raise ParseError(f"Failed to Parse the Response. {messages[-1]['content']}")
                question, answer, reasoning_steps = [s.strip() for s in matched[0]]
                reasoning_steps = list(reasoning_steps.split("\n"))
                break
            except ParseError as e:
                print(f"While exploring paper {pid}: {str(e)}")
            except ValueError as e:
                pass
            except FileNotFoundError as e:
                print(f"While exploring paper {pid}: {str(e)}")
            except Exception as e:
                print(f"Failed to explore the paper {pid}. {str(e)}")
        else:
            print(f"Failed to explore the paper after {attemps} attempts.")
            return None
        
        self.pid = explorer.pid
        
        if kwargs.get("log_dir", None):
            with open(os.path.join(kwargs["log_dir"], f"extractor_llm_call_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.json"), "w", encoding="utf-8") as f:
                json.dump(messages, f, ensure_ascii=False, indent=4)
        
        # Track the question
        tracker: SingleTracker = self.tracker_cls(model=self.model, temperature=self.temperature)
        eval_preference = kwargs.get("qa_eval_preference", "")
        messages = tracker.track(
            messages=messages,
            question=question,
            answer=answer,
            eval_preference=eval_preference,
        )
        parsed_tracker = parse_tracker_response(messages[-1]["content"])
        if parsed_tracker is None:
            print(f"Failed to Parse the Response. {messages[-1]['content']}")
            return None
        question = parsed_tracker["question"]
        answer_format = parsed_tracker["answer_format"]
        answer = parsed_tracker["answer"]
        evaluator = parse_evaluator_field(parsed_tracker["evaluator"], question=question, answer=answer)
        eval_tag = parsed_tracker["tag"]
        if evaluator is None:
            print(f"Failed to parse the evaluator. {parsed_tracker['evaluator']}")
            return None
        answer = coerce_answer_for_evaluator(answer, evaluator)
        tags.append(eval_tag)

        valid_quality, quality_reason = validate_example_quality(question, answer_format, answer)
        if not valid_quality:
            print(f"Rejected low-quality QA example: {quality_reason}")
            print(f"Question: {question}")
            print(f"Answer Format: {answer_format}")
            print(f"Answer: {answer}")
            return None
        
        try:
            if evaluate_airqa(answer, {'evaluator': evaluator}) < 0.5:
                print(f"Answer not valid.")
                print(f"Gold Answer: {answer}")
                print(f"Question: {question}")
                print(f"Answer Format: {answer_format}")
                print(f"Evaluator: {evaluator}")
                return None
        except Exception as e:
            print(f"Failed to evaluate the answer. {str(e)}")
            return None
        
        if kwargs.get("log_dir", None):
            with open(os.path.join(kwargs["log_dir"], f"extractor_llm_call_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.json"), "w", encoding="utf-8") as f:
                json.dump(messages, f, ensure_ascii=False, indent=4)
        
        example = {
            "question": question,
            "answer_format": answer_format,
            "tags": tags,
            "evaluator": evaluator,
            "annotator": self.model,
            "anchor_pdf": [self.pid]
        }
        if eval_preference:
            example["target_eval_type"] = eval_preference
        
        if explorer.exp_type == "comprehensive":
            example["conference"] = [explorer.get_conference()]

        example_dir = kwargs.get("example_dir") or EXAMPLE_DIR
        qid = generate_airqa_example_template(example_dir=example_dir, **example)["uuid"]

        concise_context = make_concise_context(context, question, answer)
        concise_reasoning = make_concise_reasoning(reasoning_steps, answer)
        metadata = {"reasoning": concise_reasoning}
        metadata.update(explorer_metadata)
        if os.fspath(example_dir) != os.fspath(EXAMPLE_DIR):
            metadata["_example_dir"] = os.fspath(example_dir)
        return qid, concise_context, answer, metadata

class MultipleAnnotator(BaseAnnotator):
    pid: List[str] = None
    def __init__(self, model: str, temperature: float, **kwargs):
        super().__init__(model, temperature, MultipleExplorer, MultipleTracker, **kwargs)
    
    def _annotate(self, **kwargs):
        self.pid = []
        forced_qids = kwargs.get("qids")
        qids = list(forced_qids) if forced_qids else random.sample(single_qids, k=2)
        if len(qids) != 2:
            raise ValueError("MultipleAnnotator requires exactly two source single question IDs.")
        examples, tags, answers = [], [], []
        questions, answer_formats = "", "You answer should be a Python list of 2 elements, each element is the answer of the corresponding question. "
        evaluator = {
            "eval_func": "eval_conjunction",
            "eval_kwargs": {
                "eval_func_list": [],
                "eval_kwargs_list": []
            }
        }
        for qid in qids:
            with open(os.path.join(EXAMPLE_DIR, f"{qid}.json"), "r", encoding="utf-8") as f:
                example = json.load(f)
            examples.append(example)
        
        for example in examples:
            self.pid.append(example["anchor_pdf"][0])
        explorer: MultipleExplorer = self.explorer_cls(pid=self.pid, model=self.model, temperature=self.temperature)
        titles = explorer.get_titles()
        
        for i, example in enumerate(examples):
            question, answer_format = example["question"], example["answer_format"]
            questions += f"In the paper \"{titles[i]}\", {question[:1].lower()}{question[1:]} "
            answer_formats += f"For question {i+1}, {answer_format[:1].lower()}{answer_format[1:]} "
            tags.extend(example["tags"])
            answers.append(example["answer"])
            evaluator["eval_kwargs"]["eval_func_list"].append(example["evaluator"]["eval_func"])
            evaluator["eval_kwargs"]["eval_kwargs_list"].append(example["evaluator"]["eval_kwargs"])
        
        tags = list(set(tags))
        tags.remove("single")
        tags.append("multiple")
        if ("objective" in tags) and ("subjective" in tags):
            tags.remove("objective")
        example = {
            "question": questions,
            "answer_format": answer_formats,
            "tags": tags,
            "evaluator": evaluator,
            "annotator": self.model,
            "anchor_pdf": self.pid,
            "from": qids
        }
        
        qid = generate_airqa_example_template(**example)["uuid"]

        context = _format_multi_field(
            examples,
            titles,
            "context",
            int(getattr(qa_config, "CONTEXT_MAX_CHARS", 900)),
        )
        reasoning = _format_multi_field(
            examples,
            titles,
            "reasoning",
            int(getattr(qa_config, "REASONING_MAX_CHARS", 500)),
        )
        supporting_examples = [
            {
                "uuid": source_qid,
                "paper_id": example.get("anchor_pdf", [""])[0] if example.get("anchor_pdf") else "",
                "context": _truncate_text(example.get("context", ""), 450),
                "reasoning": _truncate_text(example.get("reasoning", ""), 250),
            }
            for source_qid, example in zip(qids, examples)
        ]
        return qid, context, answers, {
            "reasoning": reasoning,
            "supporting_examples": supporting_examples,
        }

class RetrievalAnnotator(BaseAnnotator):
    pid: str = None
    def __init__(self, model: str, temperature: float, **kwargs):
        super().__init__(model, temperature, RetrievalExplorer, RetrievalTracker, **kwargs)
    
    def _annotate(self, **kwargs):
        attemps = EXPLORE_UUID_ATTEMP if self.pid else EXPLORE_NO_UUID_ATTEMP
        for _ in range(attemps):
            try:
                if not self.pid:
                    pid = random.choice(available_pids)
                else:
                    pid = self.pid
                explorer: RetrievalExplorer = self.explorer_cls(pid=pid, model=self.model, temperature=self.temperature)
                response = explorer.explore(**kwargs)
                pattern = r"```(?:txt)?\s*(.*?)\s*```"
                matched = re.findall(pattern, response, re.DOTALL)
                if len(matched) == 0:
                    question = response.strip()
                else:
                    question = matched[-1].strip()
                break
            except ParseError as e:
                print(f"While exploring paper {pid}: {str(e)}")
            except ValueError as e:
                pass
            except FileNotFoundError as e:
                print(f"While exploring paper {pid}: {str(e)}")
            except Exception as e:
                print(f"Failed to explore the paper {pid}. {str(e)}")
        else:
            print(f"Failed to explore the paper after {attemps} attempts.")
            return None

        example = {
            "question": question,
            "answer_format": "You answer should be a Python string, the title of the paper.",
            "tags": ['retrieval', 'objective'],
            "evaluator": {
                "eval_func": "eval_paper_relevance_with_reference_answer",
                "eval_kwargs": {
                    "question": question,
                    "reference_answer": explorer.get_title()
                }
            },
            "annotator": self.model,
            "anchor_pdf": [explorer.pid],
            "conference": [explorer.get_conference()]
        }
        
        qid = generate_airqa_example_template(**example)["uuid"]
        
        return qid, "", ""

class ComprehensiveAnnotator(SingleAnnotator):
    pid: str = None
    def __init__(self, model: str, temperature: float, **kwargs):
        super().__init__(model, temperature, ComprehensiveExplorer, ComprehensiveTracker, **kwargs)

QUESTION_TYPE_ALIASES = {
    "single": "single",
    "multi": "multi",
    "multiple": "multi",
    "rag": "rag",
    "retrieval": "rag",
    "comprehensive": "comprehensive",
}

ANNOTATORS_BY_QUESTION_TYPE = {
    "single": SingleAnnotator,
    "multi": MultipleAnnotator,
    "rag": RetrievalAnnotator,
    "comprehensive": ComprehensiveAnnotator,
}

DEFAULT_QUESTION_TYPE_ORDER = ("single", "multi", "rag", "comprehensive")


def normalize_question_type(question_type: str) -> str:
    normalized = QUESTION_TYPE_ALIASES.get(str(question_type).lower())
    if normalized is None:
        valid_types = ", ".join(sorted(ANNOTATORS_BY_QUESTION_TYPE))
        raise ValueError(f"Unknown QA question type: {question_type}. Valid types: {valid_types}.")
    return normalized


def get_type_example_counts() -> Dict[str, int]:
    raw_counts = getattr(qa_config, "TYPE_EXAMPLE_COUNTS", {"single": 1})
    counts = {question_type: 0 for question_type in ANNOTATORS_BY_QUESTION_TYPE}
    for question_type, count in dict(raw_counts).items():
        normalized = normalize_question_type(question_type)
        count = int(count)
        if count < 0:
            raise ValueError(f"QA count for {question_type} must be non-negative.")
        counts[normalized] += count
    return counts


def get_type_order(type_counts: Dict[str, int]) -> List[str]:
    order: List[str] = []
    for question_type in DEFAULT_QUESTION_TYPE_ORDER:
        normalized = normalize_question_type(question_type)
        if normalized not in order:
            order.append(normalized)

    for question_type in type_counts:
        if question_type not in order:
            order.append(question_type)
    return order


def has_required_inputs(question_type: str) -> bool:
    if question_type == "multi":
        if len(single_qids) < 2:
            print("Not enough available single questions for multi annotator.")
            return False
        print("Available Single Questions: ", len(single_qids))
        return True

    if len(available_pids) == 0:
        print("No available papers.")
        return False
    print("Available Papers: ", len(available_pids))
    return True


def save_annotated_result(result: Any) -> None:
    metadata = {}
    if isinstance(result, dict):
        qid = result["qid"]
        context = result.get("context", "")
        answer = result.get("answer", "")
        metadata = result.get("metadata", {})
    else:
        qid, context, answer = result[:3]
        if len(result) >= 4 and isinstance(result[3], dict):
            metadata = result[3]

    example_dir = metadata.pop("_example_dir", EXAMPLE_DIR)
    example_path = os.path.join(example_dir, f"{qid}.json")
    example = json.load(open(example_path, "r", encoding="utf-8"))
    example["context"] = context
    example["answer"] = answer
    example.update(metadata)
    with open(example_path, "w", encoding="utf-8") as f:
        json.dump(example, f, ensure_ascii=False, indent=4)


def run_question_type(question_type: str, target_count: int) -> int:
    annotator_cls = ANNOTATORS_BY_QUESTION_TYPE[question_type]
    cnt, fail_cnt = 0, 0

    while cnt < target_count:
        print(f"[{question_type}] Annotating #{cnt+1}/{target_count} ...")
        try:
            reset_available_uuids()
            if not has_required_inputs(question_type):
                break

            result = annotator_cls(
                model=DEFAULT_LLM_MODEL,
                temperature=DEFAULT_TEMPERATURE
            ).annotate(
                log_dir=os.fspath(qa_config.LOG_DIR),
                explore_func=qa_config.EXPLORE_FUNC
            )
            if result is None:
                fail_cnt += 1
                if fail_cnt >= qa_config.MAX_FAILURES:
                    print(f"[{question_type}] Failed after {qa_config.MAX_FAILURES} attempts.")
                    break
                continue

            save_annotated_result(result)
            cnt += 1
            fail_cnt = 0
        except Exception as e:
            print(f"[{question_type}] Failed to annotate the paper. {str(e)}")
            fail_cnt += 1
            if fail_cnt >= qa_config.MAX_FAILURES:
                print(f"[{question_type}] Failed after {qa_config.MAX_FAILURES} attempts.")
                break

    return cnt


def run_single_for_paper(pid: str, focus: str, focus_hint: str, eval_preference: str = "") -> Optional[str]:
    try:
        eval_hint = ""
        if eval_preference and str(getattr(qa_config, "EVAL_BALANCE_MODE", "soft")).lower() != "off":
            eval_hint = dict(getattr(qa_config, "EVAL_PREFERENCE_HINTS", {}) or {}).get(eval_preference, "")
        combined_hint = focus_hint
        if eval_hint:
            combined_hint = f"{focus_hint}\nEvaluation style preference: {eval_hint}"
        result = SingleAnnotator(
            pid=pid,
            model=DEFAULT_LLM_MODEL,
            temperature=DEFAULT_TEMPERATURE,
        ).annotate(
            log_dir=os.fspath(qa_config.LOG_DIR),
            explore_func="single_text",
            qa_focus=focus,
            qa_focus_hint=combined_hint,
            qa_eval_preference=eval_preference,
        )
        if result is None:
            return None
        save_annotated_result(result)
        return result["qid"] if isinstance(result, dict) else result[0]
    except Exception as exc:
        print(f"[{pid} / {focus}] Failed to annotate. {exc}")
        return None


def run_figure_for_paper(pid: str, focus_hint: str, explore_func: str | None = None) -> bool:
    try:
        explore_func = explore_func or getattr(qa_config, "FIGURE_QA_EXPLORE_FUNC", "single_image")
        result = SingleAnnotator(
            pid=pid,
            model=DEFAULT_LLM_MODEL,
            temperature=DEFAULT_TEMPERATURE,
        ).annotate(
            log_dir=os.fspath(qa_config.LOG_DIR),
            explore_func=explore_func,
            qa_focus="figure_image",
            qa_focus_hint=focus_hint,
        )
        if result is None:
            return False
        save_annotated_result(result)
        return True
    except Exception as exc:
        print(f"[{pid} / figure_image] Failed to annotate. {exc}")
        return False


def split_count_by_ratio(total: int, ratios: Dict[str, float], keys: List[str]) -> Dict[str, int]:
    total = max(0, int(total))
    cleaned = {key: max(0.0, float(ratios.get(key, 0.0))) for key in keys}
    ratio_sum = sum(cleaned.values())
    if total == 0:
        return {key: 0 for key in keys}
    if ratio_sum <= 0:
        return {key: total if index == 0 else 0 for index, key in enumerate(keys)}

    raw_counts = {key: total * value / ratio_sum for key, value in cleaned.items()}
    counts = {key: int(raw_counts[key]) for key in keys}
    remaining = total - sum(counts.values())
    ordered = sorted(keys, key=lambda key: (raw_counts[key] - counts[key], cleaned[key]), reverse=True)
    for key in ordered[:remaining]:
        counts[key] += 1
    return counts


def load_example(qid: str) -> Dict[str, Any]:
    with open(os.path.join(EXAMPLE_DIR, f"{qid}.json"), "r", encoding="utf-8") as handle:
        return json.load(handle)


def delete_example(qid: str) -> None:
    try:
        os.remove(os.path.join(EXAMPLE_DIR, f"{qid}.json"))
    except FileNotFoundError:
        pass


def example_eval_type(example: Dict[str, Any]) -> str:
    tags = set(example.get("tags", []))
    return "subjective" if "subjective" in tags else "objective"


def single_qids_by_eval_type() -> Dict[str, List[str]]:
    grouped = {"objective": [], "subjective": []}
    for qid in get_qids("single"):
        try:
            example = load_example(qid)
        except Exception:
            continue
        grouped[example_eval_type(example)].append(qid)
    for qids in grouped.values():
        random.shuffle(qids)
    return grouped


def select_eval_preference(current: Dict[str, int], target: Dict[str, int]) -> str:
    deficits = {
        key: max(0, target.get(key, 0) - current.get(key, 0))
        for key in ("objective", "subjective")
    }
    return max(
        deficits,
        key=lambda key: (deficits[key] / max(1, target.get(key, 0)), deficits[key]),
    )


def pop_qids(pool: Dict[str, List[str]], eval_type: str, count: int) -> Optional[List[str]]:
    if len(pool.get(eval_type, [])) < count:
        return None
    return [pool[eval_type].pop() for _ in range(count)]


def run_multi_from_qids(qids: List[str]) -> Optional[str]:
    try:
        result = MultipleAnnotator(
            model=DEFAULT_LLM_MODEL,
            temperature=DEFAULT_TEMPERATURE,
        ).annotate(qids=qids)
        if result is None:
            return None
        save_annotated_result(result)
        return result["qid"] if isinstance(result, dict) else result[0]
    except Exception as exc:
        print(f"[multi] Failed to create multi from {qids}. {exc}")
        return None


def run_balanced_generation() -> Dict[str, Any]:
    if getattr(qa_config, "RANDOM_SEED", None) is not None:
        random.seed(int(qa_config.RANDOM_SEED))

    pids = sorted(get_pids())
    if qa_config.MAX_PAPERS is not None:
        pids = pids[: int(qa_config.MAX_PAPERS)]
    if not pids:
        return {"error": "No available papers."}

    total_examples = getattr(qa_config, "BALANCED_TOTAL_EXAMPLES", None)
    if total_examples is None:
        total_examples = len(pids) * int(getattr(qa_config, "PER_PAPER_EXAMPLE_COUNT", 4))
    total_examples = int(total_examples)

    qtype_ratios = {
        normalize_question_type(question_type): float(ratio)
        for question_type, ratio in dict(getattr(qa_config, "QUESTION_TYPE_TARGET_RATIO", {})).items()
    }
    qtype_targets = split_count_by_ratio(
        total_examples,
        qtype_ratios,
        ["single", "multi", "rag", "comprehensive"],
    )
    eval_ratios = dict(getattr(qa_config, "EVAL_TYPE_TARGET_RATIO", {"objective": 0.7, "subjective": 0.3}))
    single_eval_targets = split_count_by_ratio(qtype_targets["single"], eval_ratios, ["objective", "subjective"])
    multi_eval_targets = split_count_by_ratio(qtype_targets["multi"], eval_ratios, ["objective", "subjective"])
    source_eval_targets = {
        "objective": (
            single_eval_targets["objective"]
            + (2 * multi_eval_targets["objective"])
            + multi_eval_targets["subjective"]
        ),
        "subjective": single_eval_targets["subjective"] + multi_eval_targets["subjective"],
    }
    source_total = sum(source_eval_targets.values())

    focus_plan = list(getattr(qa_config, "PER_PAPER_FOCUS_PLAN", []) or ["experimental_setup"])
    focus_hints = dict(getattr(qa_config, "FOCUS_HINTS", {}) or {})
    generated_source = {"objective": 0, "subjective": 0}
    attempts = 0
    max_attempts = max(source_total + int(qa_config.MAX_FAILURES), source_total * 4)

    while generated_source != source_eval_targets and attempts < max_attempts:
        eval_preference = select_eval_preference(generated_source, source_eval_targets)
        if generated_source[eval_preference] >= source_eval_targets[eval_preference]:
            break

        pid = pids[attempts % len(pids)]
        focus = focus_plan[attempts % len(focus_plan)]
        focus_hint = focus_hints.get(focus, focus)
        print(
            f"[balanced/single] {sum(generated_source.values()) + 1}/{source_total} "
            f"target={eval_preference} paper={pid} focus={focus}"
        )
        qid = run_single_for_paper(pid, focus, focus_hint, eval_preference=eval_preference)
        attempts += 1
        if not qid:
            continue

        example = load_example(qid)
        actual_eval_type = example_eval_type(example)
        if generated_source[actual_eval_type] >= source_eval_targets[actual_eval_type]:
            print(f"[balanced/single] Discarding extra {actual_eval_type} candidate {qid}.")
            delete_example(qid)
            continue
        generated_source[actual_eval_type] += 1

    reload_available_uuids()
    pool = single_qids_by_eval_type()

    reserved_final_single: Dict[str, List[str]] = {"objective": [], "subjective": []}
    for eval_type, count in single_eval_targets.items():
        reserved = pop_qids(pool, eval_type, count)
        if reserved is None:
            reserved = pop_qids(pool, eval_type, len(pool.get(eval_type, []))) or []
        reserved_final_single[eval_type].extend(reserved)

    created_multi = {"objective": 0, "subjective": 0}
    for _ in range(multi_eval_targets["objective"]):
        qids = pop_qids(pool, "objective", 2)
        if not qids:
            print("[balanced/multi] Not enough objective single candidates for an objective multi question.")
            break
        if run_multi_from_qids(qids):
            created_multi["objective"] += 1

    for _ in range(multi_eval_targets["subjective"]):
        qids = None
        subjective_qids = pop_qids(pool, "subjective", 1)
        objective_qids = pop_qids(pool, "objective", 1)
        if subjective_qids and objective_qids:
            qids = subjective_qids + objective_qids
        elif subjective_qids:
            fallback = pop_qids(pool, "subjective", 1)
            qids = subjective_qids + fallback if fallback else None
        if not qids:
            print("[balanced/multi] Not enough subjective-support candidates for a subjective multi question.")
            break
        if run_multi_from_qids(qids):
            created_multi["subjective"] += 1

    return {
        "total_examples_target": total_examples,
        "question_type_targets": qtype_targets,
        "single_eval_targets": single_eval_targets,
        "multi_eval_targets": multi_eval_targets,
        "single_source_targets": source_eval_targets,
        "single_source_generated": generated_source,
        "single_generation_attempts": attempts,
        "reserved_final_single": {key: len(value) for key, value in reserved_final_single.items()},
        "created_multi": created_multi,
    }


def run_per_paper_generation() -> Dict[str, Dict[str, int]]:
    if getattr(qa_config, "RANDOM_SEED", None) is not None:
        random.seed(int(qa_config.RANDOM_SEED))

    pids = sorted(get_pids())
    if qa_config.MAX_PAPERS is not None:
        pids = pids[: int(qa_config.MAX_PAPERS)]

    focus_plan = list(getattr(qa_config, "PER_PAPER_FOCUS_PLAN", []) or ["experimental_setup"])
    focus_hints = dict(getattr(qa_config, "FOCUS_HINTS", {}) or {})
    per_paper_count = int(getattr(qa_config, "PER_PAPER_EXAMPLE_COUNT", 4))
    figure_count = 0
    if getattr(qa_config, "ENABLE_FIGURE_QA", False):
        figure_count = max(0, int(getattr(qa_config, "FIGURE_QA_PER_PAPER_COUNT", 1)))
    figure_count = min(per_paper_count, figure_count)
    text_count = per_paper_count - figure_count
    summary: Dict[str, Dict[str, int]] = {}

    for pid in pids:
        generated = 0
        failed = 0
        for index in range(per_paper_count):
            is_figure_slot = index >= text_count
            focus = "figure_image" if is_figure_slot else focus_plan[index % len(focus_plan)]
            focus_hint = (
                getattr(qa_config, "FIGURE_QA_FOCUS_HINT", "")
                if is_figure_slot
                else focus_hints.get(focus, focus)
            )
            print(f"[{pid}] Annotating focus {focus} ({index + 1}/{per_paper_count}) ...")
            success = False
            for _ in range(max(1, int(qa_config.MAX_FAILURES))):
                if is_figure_slot:
                    success = run_figure_for_paper(
                        pid,
                        focus_hint,
                        explore_func=getattr(qa_config, "FIGURE_QA_EXPLORE_FUNC", "single_image"),
                    )
                else:
                    success = run_single_for_paper(pid, focus, focus_hint)
                if success:
                    success = True
                    generated += 1
                    break
                failed += 1
            if not success:
                print(f"[{pid}] Failed focus {focus} after {qa_config.MAX_FAILURES} attempts.")
        summary[pid] = {
            "generated": generated,
            "requested": per_paper_count,
            "text_requested": text_count,
            "figure_requested": figure_count,
            "failed_attempts": failed,
        }
    return summary


def run_from_config() -> None:
    os.makedirs(qa_config.LOG_DIR, exist_ok=True)
    prepared_count = prepare_parsed_json_inputs()
    if prepared_count:
        print(f"Prepared {prepared_count} parsed JSON paper input(s) for AirQA.")
    reload_available_uuids()

    generation_mode = str(getattr(qa_config, "GENERATION_MODE", "counts")).lower()
    if generation_mode == "balanced":
        balanced_summary = run_balanced_generation()
        type_counts = balanced_summary.get("question_type_targets", {})
        summary = {
            "single_source_generated": balanced_summary.get("single_source_generated", {}),
            "created_multi": balanced_summary.get("created_multi", {}),
        }
        print("QA balanced generation summary:")
        print(json.dumps(balanced_summary, ensure_ascii=False, indent=2))
    elif generation_mode == "per_paper":
        paper_summary = run_per_paper_generation()
        type_counts = {"single": sum(item["requested"] for item in paper_summary.values())}
        summary = {"single": sum(item["generated"] for item in paper_summary.values())}
        print("QA per-paper generation summary:")
        for pid, item in paper_summary.items():
            print(f"- {pid}: {item['generated']}/{item['requested']} generated")
    else:
        type_counts = get_type_example_counts()
        type_order = get_type_order(type_counts)
        summary: Dict[str, int] = {}

        for question_type in type_order:
            target_count = type_counts.get(question_type, 0)
            if target_count <= 0:
                summary[question_type] = 0
                continue
            summary[question_type] = run_question_type(question_type, target_count)

        print("QA generation summary:")
        for question_type in type_order:
            print(f"- {question_type}: {summary.get(question_type, 0)}/{type_counts.get(question_type, 0)}")

    try:
        from extractor.make_dataset import make_dataset

        make_dataset()
    except Exception as exc:
        print(f"Failed to build QA dataset JSONL. {exc}")

    if getattr(qa_config, "RUN_OUTPUT_DIR", None):
        pipeline_name = "qa_extractor"
        if getattr(qa_config, "PHASE", ""):
            pipeline_name = f"qa_extractor_{qa_config.PHASE}"
        record_pipeline_run(
            qa_config.RUN_OUTPUT_DIR,
            pipeline_name,
            status="completed",
            inputs={
                "paper_dir": os.fspath(qa_config.PAPER_DIR),
                "prepared_json_inputs": prepared_count,
            },
            outputs={
                "metadata_dir": os.fspath(qa_config.METADATA_DIR),
                "processed_data_dir": os.fspath(qa_config.PROCESSED_DATA_DIR),
                "example_dir": os.fspath(qa_config.EXAMPLE_DIR),
                "dataset_path": os.fspath(qa_config.OUTPUT_DATASET_PATH),
                "log_dir": os.fspath(qa_config.LOG_DIR),
            },
            extra={
                "generation_mode": generation_mode,
                "requested_counts": type_counts,
                "generated_counts": summary,
            },
        )


if __name__ == "__main__":
    run_from_config()
