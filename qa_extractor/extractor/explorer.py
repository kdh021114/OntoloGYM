#coding=utf8
from abc import ABC, abstractmethod
import os, sys, json, random, copy, string, base64, re
from pathlib import Path
from typing import List, Dict, Any, Optional
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
ONTOLOGYM_ROOT = Path(__file__).resolve().parents[2]
if os.fspath(ONTOLOGYM_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(ONTOLOGYM_ROOT))

try:
    import pymupdf
except ImportError:
    pymupdf = None

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import pdf2image
except ImportError:
    pdf2image = None

from runtime import load_config
from utils.llm_utils import call_llm_with_message, get_image_message, convert_to_message, call_llm, DEFAULT_LLM_MODEL, DEFAULT_TEMPERATURE, truncate_tokens
from extractor.explorer_prompt import EXPLORE_PROMPT, CONTEXT_PROMPT, IMAGE_PROMPT
from hysteresis_ablation.assets import load_asset_manifest
from hysteresis_ablation.prompts import HYSTERESIS_LOOP_CONTEXT

qa_config = load_config()
qa_config.ensure_directories()

DATA_DIR = os.fspath(qa_config.DATA_DIR)
PAPER_DIR = os.fspath(qa_config.PAPER_DIR)
TMP_DIR = os.fspath(qa_config.TMP_DIR)
PROCESSED_DIR = os.fspath(qa_config.PROCESSED_DATA_DIR)
METADATA_DIR = os.fspath(qa_config.METADATA_DIR)
EXAMPLE_DIR = os.fspath(qa_config.EXAMPLE_DIR)


def _resolve_pdf_path(paper_id: str, recorded_path: str | None = None) -> str:
    paper_root = Path(PAPER_DIR)
    candidates = []
    if recorded_path:
        raw_path = Path(str(recorded_path))
        candidates.extend([
            raw_path,
            Path.cwd() / raw_path,
            qa_config.ONTOLOGYM_ROOT / raw_path,
        ])
        rewritten = str(recorded_path).replace("data/dataset/airqa/papers", os.fspath(paper_root))
        candidates.append(Path(rewritten))

    candidates.append(paper_root / f"{paper_id}.pdf")
    candidates.extend(paper_root.glob(f"*/{paper_id}.pdf"))
    candidates.extend(paper_root.rglob(f"{paper_id}.pdf"))

    for candidate in candidates:
        if candidate.exists():
            return os.fspath(candidate)

    raise FileNotFoundError(f"PDF for Paper ID {paper_id} does not exist under {paper_root}.")


def _mineru_payload(pdf_data: Dict[str, Any]) -> Dict[str, Any]:
    nested_payload = pdf_data.get("info_from_mineru")
    payload = nested_payload if isinstance(nested_payload, dict) else pdf_data
    if "TOC" not in payload and isinstance(payload.get("sections"), list):
        payload = dict(payload)
        payload["TOC"] = payload["sections"]
    return payload


def section_partition(section_data: List[Dict[str, Any]]) -> List[str]:
    partitions = _section_partition_pass(section_data, require_included=True)
    if not partitions and getattr(qa_config, "INCLUDED_SECTIONS", None):
        partitions = _section_partition_pass(section_data, require_included=False)
    return partitions


def _section_partition_pass(section_data: List[Dict[str, Any]], require_included: bool) -> List[str]:
    partitions = []
    for data in section_data:
        title = str(data.get("title", "")).strip()
        text = str(data.get("text", "")).strip()
        title_lower = title.lower()
        text_lower = text.lower()
        if not _section_allowed_for_text_qa(title, require_included=require_included):
            continue
        if title_lower.startswith("references") or title_lower.startswith("acknowledg"):
            continue
        if "doi:" in text_lower and "keywords:" in text_lower:
            continue
        title_text = title + "\n" + text
        if _has_substantive_section_text(title, text):
            partitions.append(title_text)
    return partitions


def _has_substantive_section_text(title: str, text: str) -> bool:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if not cleaned:
        return False
    min_chars = int(getattr(qa_config, "MIN_SECTION_TEXT_CHARS", 260))
    min_words = int(getattr(qa_config, "MIN_SECTION_WORDS", 45))
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9_.+%μµ°×^-]*", cleaned)
    if len(cleaned) < min_chars or len(words) < min_words:
        return False
    title_lower = str(title or "").lower()
    if any(marker in title_lower for marker in ("author", "affiliation", "correspondence")):
        return False
    scientific_cues = (
        "measure", "experiment", "sample", "result", "figure", "field", "temperature",
        "magnet", "synth", "fabricat", "method", "observ", "increase", "decrease",
        "compare", "mechanism", "property", "material", "nm", "μ", "k ", " t", "%"
    )
    cue_count = sum(1 for cue in scientific_cues if cue in cleaned.lower())
    return cue_count >= 1


DEFAULT_FOCUS_SECTION_KEYWORDS = {
    "experimental_setup": [
        "experimental",
        "experiment",
        "method",
        "materials",
        "measurement",
        "sample",
        "fabrication",
        "synthesis",
        "preparation",
    ],
    "reported_result": [
        "result",
        "discussion",
        "performance",
        "characterization",
        "magnetic",
        "electrochemical",
    ],
    "isa_concept": [
        "method",
        "materials",
        "experimental",
        "measurement",
        "characterization",
        "result",
        "discussion",
    ],
    "mechanism_or_comparison": [
        "result",
        "discussion",
        "mechanism",
        "analysis",
        "comparison",
        "performance",
    ],
}

FOCUS_SECTION_KEYWORDS = dict(
    getattr(qa_config, "FOCUS_SECTION_KEYWORDS", None) or DEFAULT_FOCUS_SECTION_KEYWORDS
)


def _normalized_section_title(title: str) -> str:
    title = re.sub(r"\s+", " ", str(title or "").lower()).strip()
    title = re.sub(r"^\s*(?:section\s*)?\d+(?:\.\d+)*\.?\s*", "", title)
    return title


def _title_has_keyword(title: str, keywords: List[str]) -> bool:
    normalized = _normalized_section_title(title)
    return any(str(keyword).lower() in normalized for keyword in keywords)


def _section_allowed_for_text_qa(title: str, require_included: bool = True) -> bool:
    excluded = list(getattr(qa_config, "EXCLUDED_SECTIONS", []) or [])
    included = list(getattr(qa_config, "INCLUDED_SECTIONS", []) or [])
    if excluded and _title_has_keyword(title, excluded):
        return False
    if require_included and included and not _title_has_keyword(title, included):
        return False
    return True


def _section_title(section_text: str) -> str:
    return str(section_text or "").split("\n", 1)[0].lower()


def choose_section_for_focus(section_data: List[str], focus: str | None) -> str:
    if not section_data:
        raise ValueError("No Section Data Found.")
    keywords = FOCUS_SECTION_KEYWORDS.get(str(focus or "").lower(), [])
    if keywords:
        focused = [
            section
            for section in section_data
            if any(keyword in _section_title(section) for keyword in keywords)
        ]
        if focused:
            return random.choice(focused)
    return random.choice(section_data)

def view_image(paper_id: str, page_number: int, bounding_box: List[float] = None) -> str:
    if PyPDF2 is None or pdf2image is None:
        raise ImportError("PDF-based figure rendering requires PyPDF2 and pdf2image.")
    try:
        paper_id = str(paper_id)
    except:
        raise TypeError('Page ID must be a string.')
    try:
        page_number = int(page_number)
        assert page_number > 0
    except:
        raise TypeError('Page Number must be a positive integer.')
    if bounding_box is None or bounding_box == '':
        bounding_box = []
    try:
        assert isinstance(bounding_box, list)
        assert len(bounding_box) == 0 or len(bounding_box) == 4
        for i in range(len(bounding_box)):
            bounding_box[i] = float(bounding_box[i])
    except:
        raise TypeError('Bounding Box must be a list of 0 or 4 floats.')

    pdf_filename = _resolve_pdf_path(paper_id)
    
    with open(pdf_filename, 'rb') as fin:
        pdf_reader = PyPDF2.PdfReader(fin)
        mediabox = pdf_reader.pages[page_number - 1].mediabox
        w, h = mediabox.width, mediabox.height
    image = pdf2image.convert_from_path(pdf_filename)[page_number - 1]
    width_ratio, height_ratio = float(image.width) / float(w), float(image.height) / float(h)

    if bounding_box:
        box = copy.deepcopy(bounding_box)
        box[2] *= width_ratio
        box[3] *= height_ratio
        box[0] *= width_ratio
        box[1] *= height_ratio
        image = image.crop(box)
    file_name = ''.join(random.choices(string.ascii_letters, k=10))
    image_file = os.path.join(TMP_DIR, f'mc_{file_name}.png')
    image.save(image_file, 'PNG')
    with open(image_file, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    os.remove(image_file)
    return image_data


def _figure_caption(figure_data: Dict[str, Any]) -> str:
    return str(
        figure_data.get("figure_caption")
        or figure_data.get("caption")
        or figure_data.get("title")
        or ""
    ).strip()


def _is_figure_caption(caption: str) -> bool:
    return bool(re.match(r"^\s*(figure|fig\.?)\s*[\w\d.:-]*", str(caption or ""), re.IGNORECASE))


def _maybe_add_path(candidates: List[Path], value: Any) -> None:
    if value in (None, ""):
        return
    path = Path(str(value))
    candidates.append(path)
    if not path.is_absolute():
        candidates.append(Path.cwd() / path)
        candidates.append(qa_config.ONTOLOGYM_ROOT / path)


def _existing_file(candidates: List[Path]) -> Optional[Path]:
    seen = set()
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except FileNotFoundError:
            resolved = candidate.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists() and resolved.is_file():
            return resolved
    return None


def _figure_index_candidates(caption: str) -> List[str]:
    matches = re.findall(r"(?:figure|fig\.?)\s*([sS]?\d+[A-Za-z]?)", caption or "", re.IGNORECASE)
    keys = []
    for match in matches:
        normalized = match.lower()
        keys.extend(
            [
                f"figure_{normalized}",
                f"figure{normalized}",
                f"fig_{normalized}",
                f"fig{normalized}",
            ]
        )
    return keys


def _resolve_figure_image_path(pdf_data: Dict[str, Any], figure_data: Dict[str, Any]) -> Optional[Path]:
    candidates: List[Path] = []
    paper_dir = Path(str(pdf_data.get("paper_dir") or ""))
    source_file = Path(str(pdf_data.get("source_file") or ""))
    assets = pdf_data.get("paper_assets") if isinstance(pdf_data.get("paper_assets"), dict) else {}
    figures_dir = Path(str(assets.get("figures_dir") or "")) if assets.get("figures_dir") else None

    for key in ("figure_filename", "figure_path", "image_path", "path", "filename", "file"):
        raw_value = figure_data.get(key)
        _maybe_add_path(candidates, raw_value)
        if raw_value:
            raw_path = Path(str(raw_value))
            if paper_dir:
                candidates.append(paper_dir / raw_path)
            if source_file:
                candidates.append(source_file.parent / raw_path)
            if figures_dir:
                candidates.append(figures_dir / raw_path.name)

    existing = _existing_file(candidates)
    if existing:
        return existing

    figure_paths = [
        Path(str(path))
        for path in assets.get("figure_paths", [])
        if str(path or "").strip()
    ]
    if not figure_paths and figures_dir and figures_dir.exists():
        figure_paths = [path for path in figures_dir.iterdir() if path.is_file()]
    if not figure_paths:
        return None

    explicit_name = str(figure_data.get("figure_filename") or "").strip().lower()
    if explicit_name:
        explicit_stem = Path(explicit_name).stem.lower()
        for path in figure_paths:
            if path.name.lower() == Path(explicit_name).name.lower() or path.stem.lower() == explicit_stem:
                return path if path.is_absolute() else (paper_dir / path)

    caption_keys = _figure_index_candidates(_figure_caption(figure_data))
    for key in caption_keys:
        for path in figure_paths:
            if key in path.stem.lower().replace("-", "_"):
                return path if path.is_absolute() else (paper_dir / path)
    return None


def _hysteresis_assets_for_paper(paper_id: str) -> List[Dict[str, Any]]:
    manifest_path = getattr(qa_config, "HYSTERESIS_ASSET_MANIFEST_JSONL", None)
    if not manifest_path:
        return []
    assets = load_asset_manifest(manifest_path)
    selected = [
        asset
        for asset in assets
        if str(asset.get("paper_id") or "") == str(paper_id)
        and str(asset.get("image_path") or "").strip()
    ]
    return selected


class BaseExplorer(ABC):
    model: str = None
    temperature: float = None
    
    def __init__(self, model: str, temperature: float):
        self.model = model
        self.temperature = temperature
        
    def _explore_with_llm(
            self,
            template: str,
            **kwargs
        ) -> List[Any]:
        messages = convert_to_message(template, **kwargs)
        response = call_llm_with_message(messages, model=self.model, temperature=self.temperature)
        messages.append({"role": "assistant", "content": response})
        return messages
    
    @abstractmethod
    def explore(self, **kwargs) -> Any:
        pass

class SingleExplorer(BaseExplorer):
    """Single Document Explorer.
    """
    exp_type: str = "single"
    pid: str = None
    pdf_data: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    page_data: List[str] = None
    
    def __init__(self, pid: str, model: str, temperature: float):
        super().__init__(model=model, temperature=temperature)
        self.pid = pid
        pdf_data_path = os.path.join(PROCESSED_DIR, f"{self.pid}.json")
        if not os.path.exists(pdf_data_path):
            raise FileNotFoundError(f"Processed Data File {pdf_data_path} not found.")
        with open(pdf_data_path, "r", encoding="utf-8") as f:
            self.pdf_data = json.load(f)
        
        metadata_path = os.path.join(METADATA_DIR, f"{self.pid}.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata File {metadata_path} not found.")
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        
        self.page_data = []
        try:
            pdf_path = _resolve_pdf_path(self.pid, self.pdf_data.get("pdf_path"))
        except FileNotFoundError:
            pdf_path = None
        if pdf_path and pymupdf is not None:
            doc = pymupdf.open(pdf_path)
            for page_number in range(doc.page_count):
                page = doc[page_number]
                text = page.get_text()
                self.page_data.append(text)
            doc.close()
    
    def get_title(self) -> str:
        return self.metadata["title"]
    
    def get_conference(self) -> str:
        return str(self.metadata["conference"]).lower() + str(self.metadata["year"])
    
    def get_abstract(self) -> str:
        return self.metadata["abstract"]
    
    def get_titles(self) -> List[str]:
        return [self.metadata["title"]]
    
    def explore(self, **kwargs) -> Any:
        explore_funcs = [
            "single_text",
            "single_table",
            "single_image",
            "hysteresis_image",
            "single_formula",
            "multiple_section_subsection",
            "multiple_section_section",
        ]
        explore_func = kwargs.get("explore_func", None)
        if not explore_func: explore_func=random.choice(explore_funcs)
        assert explore_func in explore_funcs, f"Invalid Explore Function {explore_func}."
        return getattr(self, explore_func)(**kwargs)
    
    def single_text(self, **kwargs) -> Any:
        """Single-Step Paradigm: Text Modal.
        """
        context_type = kwargs.get("context", "section")
        if context_type == "section":
            section_data = section_partition(_mineru_payload(self.pdf_data).get("TOC", []))
            content = choose_section_for_focus(section_data, kwargs.get("qa_focus"))
        elif context_type == "page":
            if not self.page_data:
                raise ValueError("No PDF page text is available for page-level QA generation.")
            content = random.choice(self.page_data)
        else:
            raise ValueError(f"Invalid Context Type {context_type}.")
        focus_hint = kwargs.get("qa_focus_hint", "")
        focus_block = f"\nQuestion focus: {focus_hint}\n" if focus_hint else ""
        title_block = f"\nPaper identifier: {self.pid}\nSection title: {_section_title(content)}\n"
        template = (
            EXPLORE_PROMPT[self.exp_type][context_type]
            + focus_block
            + title_block
            + CONTEXT_PROMPT[self.exp_type][context_type].format(content=truncate_tokens(content, max_tokens=5))
        )
        return self._explore_with_llm(template), content, [self.exp_type, "text"]

    def single_table(self, **kwargs) -> Any:
        """Single-Step Paradigm: Table Modal.
        """
        table_data = _mineru_payload(self.pdf_data).get("tables", [])
        if not table_data:
            raise ValueError("No Table Data Found.")
        table_data = random.choice(table_data)
        context = CONTEXT_PROMPT[self.exp_type]["table"].format(
            caption = table_data["table_caption"],
            content = truncate_tokens(table_data["table_html"], max_tokens=5)
        )
        template = EXPLORE_PROMPT[self.exp_type]["table"] + context
        return self._explore_with_llm(template), table_data["table_caption"] + '\n' + table_data["table_html"], [self.exp_type, "table"]
    
    def single_image(self, **kwargs) -> Any:
        """Single-Step Paradigm: Image Modal.
        """
        payload = _mineru_payload(self.pdf_data)
        figures = [data for data in payload.get("figures", []) if isinstance(data, dict)]
        image_data = [
            data 
            for data in figures
            if _is_figure_caption(_figure_caption(data))
        ]
        if not image_data:
            image_data = figures
        if not image_data:
            raise ValueError("No Image Data Found.")
        image_data = random.choice(image_data)
        caption = _figure_caption(image_data)
        focus_hint = kwargs.get("qa_focus_hint", "")
        focus_block = f"\nQuestion focus: {focus_hint}\n" if focus_hint else ""
        context = CONTEXT_PROMPT[self.exp_type]["image"].format(caption=caption)
        template = EXPLORE_PROMPT[self.exp_type]["image"] + focus_block + context
        image_path = _resolve_figure_image_path(self.pdf_data, image_data)
        if image_path:
            image_template = get_image_message(
                IMAGE_PROMPT[self.exp_type],
                image_path=os.fspath(image_path),
                image_limit=1,
            )
            context_text = f"{caption}\nImage file: {image_path}"
        else:
            if "page_number" not in image_data:
                raise FileNotFoundError("No figure image file or PDF page number was found for image QA.")
            image_base64 = view_image(
                paper_id=self.pid,
                page_number=image_data["page_number"],
                bounding_box=image_data.get("figure_bbox"),
            )
            image_template = get_image_message(
                IMAGE_PROMPT[self.exp_type],
                base64_image=image_base64,
                mine_type='image/png'
            )
            context_text = caption
        return self._explore_with_llm(template, image=image_template), context_text, [self.exp_type, "image"]

    def hysteresis_image(self, **kwargs) -> Any:
        """Single-step QA generation from preselected hysteresis-loop assets."""
        assets = _hysteresis_assets_for_paper(self.pid)
        if not assets:
            raise ValueError(f"No hysteresis-loop figure assets found for paper {self.pid}.")
        asset = random.choice(assets)
        caption = str(asset.get("caption") or "").strip()
        image_path = str(asset.get("image_path") or "").strip()
        if not image_path:
            raise FileNotFoundError(f"Hysteresis asset {asset.get('asset_id')} has no copied image path.")
        focus_hint = kwargs.get("qa_focus_hint", "") or getattr(qa_config, "HYSTERESIS_QA_FOCUS_HINT", "")
        figure_label = str(asset.get("figure_label") or "Hysteresis figure")
        focus_block = (
            "\nHysteresis-loop domain context:\n"
            f"{HYSTERESIS_LOOP_CONTEXT.strip()}\n\n"
            f"Question focus: {focus_hint}\n"
            "Ask a question that needs the image, not only general domain knowledge.\n"
        )
        context = CONTEXT_PROMPT[self.exp_type]["image"].format(caption=caption)
        title_block = f"\nPaper identifier: {self.pid}\nFigure: {figure_label}\n"
        template = EXPLORE_PROMPT[self.exp_type]["image"] + focus_block + title_block + context
        image_template = get_image_message(
            IMAGE_PROMPT[self.exp_type],
            image_path=image_path,
            image_limit=1,
        )
        context_text = "\n".join(
            [
                f"Hysteresis figure: {figure_label}",
                f"Caption: {caption}",
                f"Image file: {image_path}",
            ]
        )
        metadata = {
            "figure_asset_id": asset.get("asset_id", ""),
            "figure_label": figure_label,
            "figure_caption": caption,
            "figure_image_path": image_path,
            "context_source_type": "hysteresis_figure_image",
        }
        return self._explore_with_llm(template, image=image_template), context_text, [self.exp_type, "image", "hysteresis_figure"], metadata
    
    def single_formula(self, **kwargs) -> Any:
        """Single-Step Paradigm: Formula Modal.
        """
        formula_data = _mineru_payload(self.pdf_data).get("equations", [])
        if not formula_data:
            raise ValueError("No Formula Data Found.")
        index = random.randint(0, len(formula_data) - 1)
        formula_data = formula_data[index]
        context = CONTEXT_PROMPT[self.exp_type]["formula"].format(
            index=index+1,
            formula=formula_data["equation_text"]
        )
        template = EXPLORE_PROMPT[self.exp_type]["formula"] + context
        return self._explore_with_llm(template), formula_data["equation_text"], [self.exp_type, "formula"]

    def multiple_section_subsection(self, **kwargs) -> Any:
        """Multiple-Step Paradigm: Section-Subsection Modal.
        """
        section_data = section_partition(_mineru_payload(self.pdf_data).get("TOC", []))
        if not section_data:
            raise ValueError("No Section Data Found.")
        content = random.choice(section_data)
        template = EXPLORE_PROMPT[self.exp_type]["sec_sub"] + CONTEXT_PROMPT[self.exp_type]["sec_sub"].format(content=content)
        return self._explore_with_llm(template), content, [self.exp_type, "text"]

    def multiple_section_section(self, **kwargs) -> Any:
        """Multiple-Step Paradigm: Section-Section Modal.
        """
        section_data = section_partition(_mineru_payload(self.pdf_data).get("TOC", []))
        if len(section_data) < 2:
            raise ValueError("At least two sections are required.")
        indexs = sorted(random.sample(list(range(0, len(section_data))), 2))
        section_data = [section_data[index].strip() for index in indexs]
        context = CONTEXT_PROMPT[self.exp_type]["sec_sec"].format(content0=section_data[0], content1=section_data[1])
        template = EXPLORE_PROMPT[self.exp_type]["sec_sec"] + context
        return self._explore_with_llm(template), section_data[0] + "\n" + section_data[1], [self.exp_type, "text"]

class MultipleExplorer(BaseExplorer):
    """Multiple Document Explorer.
    """
    exp_type: str = "multiple"
    pid: List[str] = None
    subexplorers: List[SingleExplorer] = []
    def __init__(self, pid: List[str], model: str, temperature: float):
        super().__init__(model=model, temperature=temperature)
        self.pid = pid
        self.subexplorers = [SingleExplorer(pid=_pid, model=model, temperature=temperature) for _pid in pid]
    
    def get_conference(self):
        return None
    
    def get_titles(self) -> List[str]:
        return [explorer.get_title() for explorer in self.subexplorers]

    def explore(self, **kwargs) -> None:
        pass

RETRIEVAL_PROMPT = """You are an intelligent annotation system who is expert in posing questions. You need to pose a question based on the title and abstract of a paper, where the answer to the question should be the title of the paper. That is to say, you need to describte the contribution or the feature of the paper in the question, so that the respondents can identify the paper. Don't include the title itself in the question. Now let's start!

[Title]: {title}
[Abstract]: {abstract}

Your output should be in the following format:

Your thought process.
```txt
Your question here.
```

Note that, you should wrap your output with triple backticks.
"""

class RetrievalExplorer(SingleExplorer):
    """Retrieval Document Explorer.
    """
    exp_type: str = "retrieval"
    pid: str = None
    def __init__(self, pid: str, model: str, temperature: float):
        super().__init__(pid=pid, model=model, temperature=temperature)
    
    def explore(self, **kwargs) -> Any:
        return call_llm(
            RETRIEVAL_PROMPT.format(
                title=self.get_title(),
                abstract=self.get_abstract()
            ),
            model=self.model,
            temperature=self.temperature
        )

class ComprehensiveExplorer(SingleExplorer):
    """Comprehensive Document Explorer.
    """
    exp_type: str = "comprehensive"
    def _explore_with_llm(
            self,
            template: str,
            **kwargs
        ) -> List[Any]:
        template += "\nThe title of the paper is as follows:\n```txt\n{title}\n```\nThe abstract of the paper is as follows:\n```markdown\n{abstract}\n```\n".format(title=self.get_title(), abstract=self.get_abstract())
        return super()._explore_with_llm(template, **kwargs)
