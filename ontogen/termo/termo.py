from __future__ import annotations

"""
Termo: A class for extracting and processing domain-specific terminology from a text.

This module supports term extraction, acronym expansion, definition identification,
and relationship extraction using LLMs (via Ollama, Anthropic, or OpenAI). It includes
post-processing to remove duplicates, substrings, and hallucinated terms.
"""
import os
import re
from functools import cmp_to_key

try:
    import anthropic
except ImportError:  # pragma: no cover - optional runtime dependency.
    anthropic = None

try:
    import ollama
except ImportError:  # pragma: no cover - optional runtime dependency.
    ollama = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional runtime dependency.
    OpenAI = None

from common.usage_logging import log_openai_usage

try:
    import spacy
except ImportError:  # pragma: no cover - optional runtime dependency.
    spacy = None

try:
    from prompt import (
        prompt_abstract,
        prompt_acronym,
        prompt_definitions,
        prompt_evidence_terms,
        prompt_evidence_triplets,
        prompt_table_terms,
        prompt_table_triplets,
        prompt_triplets,
    )
except ImportError:  # pragma: no cover - package import path.
    from .prompt import (
        prompt_abstract,
        prompt_acronym,
        prompt_definitions,
        prompt_evidence_terms,
        prompt_evidence_triplets,
        prompt_table_terms,
        prompt_table_triplets,
        prompt_triplets,
    )

try:
    from thinc.api import require_gpu, set_gpu_allocator
except ImportError:  # pragma: no cover - optional runtime dependency.
    def require_gpu(*args, **kwargs):
        return None

    def set_gpu_allocator(*args, **kwargs):
        return None


VALID_CONTEXT_KINDS = {"generic", "evidence", "table", "enriched"}
VALID_RELATIONSHIP_MODES = {"generic", "evidence"}

TERM_SECTION_HEADING_PATTERNS = [
    "domain terms",
    "evidence terms",
    "table-derived",
    "figure-derived",
    "experimental settings",
    "numeric values",
    "methods /",
    "quantities /",
]

TERM_CLAUSE_MARKERS = {
    " is ",
    " are ",
    " was ",
    " were ",
    " be ",
    " been ",
    " being ",
    " shows ",
    " show ",
    " indicates ",
    " induces ",
    " indicate ",
    " suggests ",
    " suggest ",
    " causes ",
    " cause ",
    " leads to ",
    " related to ",
    " due to ",
}

GENERIC_SINGLE_WORD_TERMS = {
    "behavior",
    "composite",
    "data",
    "dependency",
    "dependencies",
    "effect",
    "evidence",
    "experiment",
    "figure",
    "measurement",
    "method",
    "model",
    "process",
    "region",
    "regions",
    "result",
    "results",
    "sample",
    "signals",
    "system",
    "value",
    "values",
}

GENERIC_PHRASE_TERMS = {
    "paired signals",
}


class Termo(dict):
    """
    Extract and manage terms, acronyms, relationships, and definitions from text.

    Inherits from `dict`, storing results under keys like "terms", "acronyms", 
    "relationships", and "definitions".
    """
    def __init__(
        self,
        text,
        remove_duplicates=True,
        remove_substrings=True,
        backend="ollama",
        context_kind="generic",
        relationship_mode="generic",
        include_literal_values=False,
        preserve_table_blocks=False,
    ):
        """
        text: str
            The text from which to extract terms, acronyms, relationships and definitions
        remove_duplicates: bool
            Whether to remove duplicated terms from the list of identified terms
        remove_substrings: bool
            Whether to remove terms that are substrings of other terms
            backend: str
            The backend to use for the queries. Available backends are
            'ollama', 'anthropic', and 'openai'/'oai'.
        context_kind: str
            Extraction context mode: 'generic', 'evidence', 'table', or 'enriched'.
        relationship_mode: str
            Relationship prompt mode: 'generic' or 'evidence'.
        include_literal_values: bool
            Whether relationship objects may be literal values explicitly present in the context.
        preserve_table_blocks: bool
            Whether [TABLE ...] blocks should be preserved during chunking when possible.
        """
        self.text = text
        self.remove_duplicates = remove_duplicates
        self.remove_substrings = remove_substrings
        self.context_kind = context_kind
        self.relationship_mode = relationship_mode
        self.include_literal_values = include_literal_values
        self.preserve_table_blocks = preserve_table_blocks
        if self.context_kind not in VALID_CONTEXT_KINDS:
            raise ValueError(
                f"Unknown context_kind: '{context_kind}'. Expected one of {sorted(VALID_CONTEXT_KINDS)}"
            )
        if self.relationship_mode not in VALID_RELATIONSHIP_MODES:
            raise ValueError(
                f"Unknown relationship_mode: '{relationship_mode}'. Expected one of {sorted(VALID_RELATIONSHIP_MODES)}"
            )
        self["terms"] = []
        self["acronyms"] = {}
        if backend == "ollama":
            self.query_fn = self.query_ollama
        elif backend == "anthropic":
            self.query_fn = self.query_anthropic
        elif backend in {"openai", "oai"}:
            self.query_fn = self.query_openai
        else:
            raise ValueError(
                f"Unknown backend: '{backend}'. Available backends are 'ollama', 'anthropic', and 'openai'"
            )

    def extract_terms(
        self, model, space_separator=True, max_length_split=2000, remove_hallucinated=True, **kwargs
    ):
        """
        Extract terms from the text using the specified model
        model: str
            The model to use for the queries
        space_separator: bool
            If True, only terms separated by spaces are considered. If False, any match of the given term in the text is considered,
            even if its not surrounded by spaces.
        max_length_split: int
            The maximum length of the text to send in each query in characters
        """
        self["terms"] += self.get_filtered_list_from_llm(
            model, self.text, space_separator, max_length_split, remove_hallucinated, **kwargs
        )
        self["terms"] = self.postprocess_terms(
            self["terms"], self.remove_duplicates, self.remove_substrings
        )
        return self["terms"]

    def extract_acronyms(self, model, max_length_split=2000, **kwargs):
        """
        Extract acronyms and map them to their full forms using LLM.

        Args:
            model (str): Model identifier.
            max_length_split (int): Max characters per query.
            **kwargs: Additional keyword arguments for LLM query.

        Returns:
            dict: Mapping of acronym -> full form.
        """
        terms = [term[0] for term in self["terms"]]
        self["acronyms"] = self.get_acronyms_from_llm(
            model, self.text, terms, max_length_split, **kwargs
        )
        self["acronyms"] = self.postprocess_acronyms(
            self["acronyms"], self.text, terms
        )
        return self["acronyms"]

    def extract_relationships(self, model, max_length_split=2000, **kwargs):
        """
        Extract relationships between terms and acronyms using LLM.

        Args:
            model (str): Model identifier.
            max_length_split (int): Max characters per LLM query.
            **kwargs: Additional keyword arguments for query function.

        Returns:
            list: List of triplets (term1, relationship, term2).
        """
        terms = [term[0] for term in self["terms"]]
        acronyms = [ac[0] for ac in self["acronyms"].items()]
        # here, exclusively, we consider acronyms as terms
        terms += acronyms
        self["relationships"] = self.get_relationships_from_llm(
            model, self.text, terms, max_length_split, **kwargs
        )
        self["relationships"] = self.postprocess_relationships(
            self["relationships"], self.text, terms
        )
        return self["relationships"]

    def extract_definitions(self, model, max_length_split=2000, **kwargs):
        """
        Extract definitions for identified terms using LLM.

        Args:
            model (str): Model name to query.
            max_length_split (int): Max characters per chunk sent to LLM.
            **kwargs: Extra arguments for query.

        Returns:
            dict: Mapping of term -> definition.
        """
        terms = [term[0] for term in self["terms"]]
        self["definitions"] = self.get_definitions_from_llm(
            model, self.text, terms, max_length_split, **kwargs
        )
        self["definitions"] = self.postprocess_definitions(
            self["definitions"], self.text, terms
        )
        return self["definitions"]

    def query_anthropic(self, model, prompt, **kwargs):
        """
        Query the Anthropic API with a given prompt.

        Args:
            model (str): Anthropic model name.
            prompt (str): Prompt to send.
            **kwargs: Extra parameters (ignored here).

        Returns:
            str: Textual response from the model.
        """
        if anthropic is None:
            raise ImportError("anthropic is required for the Anthropic backend.")
        client = anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"],
        )
        message = client.messages.create(
            model=model,
            max_tokens=8000,
            temperature=0.3,
            messages=[
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ],
        )
        return message.content[0].text

    def query_ollama(self, model, prompt, **kwargs):
        """
        Query Ollama with a given prompt.

        Args:
            model (str): Ollama model name.
            prompt (str): Prompt to send.
            **kwargs: Extra parameters passed to Ollama.

        Returns:
            str: Response string from the model.
        """
        if ollama is None:
            raise ImportError("ollama is required for the Ollama backend.")
        response = ollama.generate(model=model, prompt=prompt, **kwargs)
        return response["response"]

    def query_openai(self, model, prompt, **kwargs):
        """
        Query the OpenAI Chat Completions API with a given prompt.
        """
        if OpenAI is None:
            raise ImportError("openai is required for the OpenAI backend.")

        options = dict(kwargs.get("options", {}) or {})
        base_url = options.pop("base_url", None) or os.getenv("OPENAI_BASE_URL")
        api_key = options.pop("api_key", None) or os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key, base_url=base_url)

        allowed_options = {
            "temperature",
            "top_p",
            "max_completion_tokens",
            "reasoning_effort",
            "verbosity",
            "seed",
        }
        if "max_tokens" in options and "max_completion_tokens" not in options:
            options["max_completion_tokens"] = options.pop("max_tokens")
        api_options = {
            key: value
            for key, value in options.items()
            if key in allowed_options and value is not None
        }
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **api_options,
        )
        log_openai_usage(completion, component="ontogen_termo")
        return completion.choices[0].message.content or ""

    def postprocess_terms(
        self, all_terms, remove_duplicates=True, remove_substrings=True
    ):
        """
        Apply duplicate and substring filtering to term list.

        Args:
            all_terms (list): List of term tuples.
            remove_duplicates (bool): Whether to deduplicate.
            remove_substrings (bool): Whether to remove substrings.

        Returns:
            list: Cleaned list of term tuples.
        """
        all_terms = self.filter_ontology_node_terms(all_terms)
        if remove_duplicates:
            all_terms = self.remove_duplicated_terms(all_terms)
        if remove_substrings:
            excluded_terms = [x for x in all_terms if x[1] == -1] # hallucinated terms are not filtered out
            all_terms = [x for x in all_terms if x[1] != -1] # any other term
            all_terms = self.remove_substrings_from_list(all_terms)
            all_terms += excluded_terms
        return all_terms

    def filter_ontology_node_terms(self, all_terms):
        """
        Remove strings that are useful as text spans but poor ontology node labels.
        """
        filtered_terms = []
        for term_tuple in all_terms:
            term = term_tuple[0] if isinstance(term_tuple, tuple) else str(term_tuple)
            normalized_term = self._normalize_term_label(term)
            if not self._is_good_ontology_node(normalized_term):
                continue
            if isinstance(term_tuple, tuple):
                filtered_terms.append((normalized_term, *term_tuple[1:]))
            else:
                filtered_terms.append(normalized_term)
        return filtered_terms

    def _normalize_term_label(self, term):
        term = str(term or "").strip()
        term = term.replace("**", "")
        term = term.strip("`*_ \t\n\r")
        term = re.sub(r"^\(?[a-zA-Z0-9]+\)?[:.)]\s+", "", term)
        term = re.sub(r"\s+", " ", term)
        return term.strip()

    def _is_good_ontology_node(self, term):
        if not term:
            return False
        lower_term = term.lower()
        if any(pattern in lower_term for pattern in TERM_SECTION_HEADING_PATTERNS):
            return False
        if lower_term in GENERIC_PHRASE_TERMS:
            return False
        if any(marker in lower_term for marker in TERM_CLAUSE_MARKERS):
            return False
        if re.fullmatch(r"(fig(?:ure)?|table|eq(?:uation)?)\.?\s*\d+[a-z]?", lower_term):
            return False
        if re.search(r"\((?:fig(?:ure)?|table|eq(?:uation)?)\.?\s*\d+[a-z]?\)", lower_term):
            return False
        if re.fullmatch(r"(left|right|upper|lower|middle|top|bottom)(?:\s+panel)?", lower_term):
            return False
        if lower_term.startswith(("end of ", "lack of ", "presence of ", "absence of ")):
            return False
        if "arrow" in lower_term or "caption" in lower_term or "see text" in lower_term:
            return False
        if " around " in lower_term:
            return False
        if term.endswith(".") or term.endswith(":"):
            return False
        if len(term) > 96:
            return False
        words = re.findall(r"[A-Za-z0-9.+/%\[\]-]+", term)
        if len(words) > 6:
            return False
        if len(words) == 1 and lower_term in GENERIC_SINGLE_WORD_TERMS:
            return False
        if re.fullmatch(r"[-+−]?\d+(?:\.\d+)?\s*(?:k?oe|k|nm|pa|ma|kv|g/cm3|%)", lower_term):
            return False
        return True

    def postprocess_relationships(self, relationships, text, terms):
        """
        Remove those relationships with terms not in terms list
        """
        result = []
        lower_terms = [term.lower() for term in terms]
        for t1, rel, t2 in relationships:
            subject_is_term = t1.lower() in lower_terms
            object_is_term = t2.lower() in lower_terms
            object_is_literal = self._is_allowed_literal_object(t2, text)
            if subject_is_term and (object_is_term or object_is_literal):
                result.append((t1, rel, t2))
            else:
                print(
                    f"Removing relationship '{t1} > {rel} > {t2}' because it contains terms not in the text"
                )
        return result

    def _is_allowed_literal_object(self, value, text):
        if not self.include_literal_values:
            return False
        normalized_value = str(value).strip()
        if not normalized_value:
            return False
        if normalized_value.lower() not in text.lower():
            return False
        return self._looks_like_literal(normalized_value)

    def _looks_like_literal(self, value):
        return bool(
            re.search(r"\d", value)
            or re.search(r"\b(?:ms|s|sec|min|h|kg|g|mg|ug|m|cm|mm|um|nm|K|C|F|Hz|kHz|MHz|GHz|Pa|bar|V|A|W|J|mol|M|mM|uM|%|percent)\b", value)
        )

    def postprocess_definitions(self, definitions, text, terms):
        """
        Remove those relationships with terms not in terms list
        """
        result = {}
        lower_terms = [term.lower() for term in terms]
        for term, defi in definitions.items():
            if term.lower() in lower_terms:
                result[term] = defi
            else:
                print(
                    f"Removing definitions for '{term}' because unknown term"
                )
        return result

    def remove_duplicated_terms(self, all_terms):
        """
        Remove duplicated term entries.

        Args:
            all_terms (list): List of term tuples or strings.

        Returns:
            list: Deduplicated list.
        """
        deduped = []
        seen = set()
        for term in all_terms:
            label = term[0] if isinstance(term, tuple) else term
            key = self._normalize_term_label(label).lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(term)
        return deduped

    def remove_substrings_from_list(self, all_terms):
        """
        Sort by increasing end position. If two terms have the same end position, sort by increasing start position
        """
        def comp(x, y):
            if x[1] != y[1]:  # compare start position
                return x[1] - y[1]
            return y[2] - x[2]  # compare end position

        all_terms = sorted(all_terms, key=cmp_to_key(comp))

        # remove all terms that are substrings of other terms
        if len(all_terms) == 0:
            return []
        start, end = all_terms[0][1], all_terms[0][2]
        result = [all_terms[0]]
        for i in range(1, len(all_terms)):
            if all_terms[i][1] >= start and all_terms[i][2] <= end:
                print(
                    f"Removing term {all_terms[i][0]} because it is a substring of {all_terms[i-1][0]}"
                )
                continue
            start, end = all_terms[i][1], all_terms[i][2]
            result.append(all_terms[i])

        return result

    def split_text_into_lines(self, text):
        """
        Segment input text into sentences using spaCy.

        Args:
            text (str): Text to split.

        Returns:
            list: List of sentences.
        """
        if spacy is not None:
            try:
                set_gpu_allocator("pytorch")
                require_gpu(0)
                nlp = spacy.load("en_core_web_trf")
                doc = nlp(text)
                return [sent.text for sent in doc.sents]
            except (OSError, ValueError, RuntimeError):
                pass

        sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
        return [sentence for sentence in sentences if sentence.strip()]

    def get_list_from_llm(self, model, text, max_length_split=2000, **kwargs):
        """
        Extract raw list of terms from text using LLM via chunked prompts.

        Args:
            model (str): Model name.
            text (str): Text to analyze.
            max_length_split (int): Max characters per chunk.
            **kwargs: Extra parameters for LLM.

        Returns:
            list: List of raw term strings.
        """
        list_terms = []
        chunks = self._build_chunks(text, max_length_split)

        for chunk in chunks:
            if len(chunk.replace("\n", "").strip()) == 0:
                continue
            prompt = self._term_prompt().format(text=chunk)
            response = self.query_fn(model, prompt, **kwargs)
            for l in response.split("\n"):
                if len(l) > 0 and l[0] == "-" and len(l[1:].strip()) > 0:
                    term = self._normalize_term_label(l[1:].strip())
                    if self._is_good_ontology_node(term):
                        list_terms.append(term)

        return list_terms

    def _term_prompt(self):
        if self.context_kind == "generic":
            return prompt_abstract
        if self.context_kind == "table":
            return prompt_table_terms
        return prompt_evidence_terms

    def _relationship_prompt(self):
        if self.context_kind == "generic" and self.relationship_mode == "generic":
            return prompt_triplets
        if self.context_kind == "table":
            return prompt_table_triplets
        return prompt_evidence_triplets

    def _build_chunks(self, text, max_length_split):
        """
        Build chunks of text for LLM processing.
        """
        if self.preserve_table_blocks or self.context_kind in {"table", "enriched"}:
            return self._build_table_aware_chunks(text, max_length_split)

        lines = self.split_text_into_lines(text)
        chunks = []
        current_chunk = ""
        for line in lines:
            line = line.strip()
            if len(current_chunk) + len(line) > max_length_split:
                chunks.append(current_chunk)
                current_chunk = ""
            current_chunk = current_chunk + " " + line
        if len(current_chunk) > 0:
            chunks.append(current_chunk)
        return chunks

    def _split_text_blocks_for_tables(self, text):
        blocks = []
        current = []
        current_is_table = False

        for line in text.splitlines():
            is_table_start = line.strip().startswith("[TABLE ")
            if is_table_start:
                if current:
                    blocks.append((current_is_table, "\n".join(current).strip()))
                current = [line]
                current_is_table = True
                continue

            if current_is_table and line.strip() == "":
                current.append(line)
                blocks.append((True, "\n".join(current).strip()))
                current = []
                current_is_table = False
                continue

            current.append(line)

        if current:
            blocks.append((current_is_table, "\n".join(current).strip()))
        return [(is_table, block) for is_table, block in blocks if block]

    def _split_table_block(self, block, max_length_split):
        if len(block) <= max_length_split:
            return [block]

        lines = block.splitlines()
        header = []
        rows = []
        in_rows = False
        for line in lines:
            if line.strip() == "Rows:":
                in_rows = True
                header.append(line)
                continue
            if in_rows and line.startswith("- "):
                rows.append(line)
            else:
                header.append(line)

        if not rows:
            return self._split_long_text(block, max_length_split)

        chunks = []
        header_text = "\n".join(header).strip()
        current_chunk = header_text
        for row in rows:
            separator = "\n" if current_chunk else ""
            if len(current_chunk) + len(separator) + len(row) > max_length_split:
                if current_chunk.strip() and current_chunk != header_text:
                    chunks.append(current_chunk)
                current_chunk = header_text + "\n" + row
            else:
                current_chunk = current_chunk + separator + row
        if current_chunk.strip():
            chunks.append(current_chunk)
        return chunks

    def _split_long_text(self, text, max_length_split):
        chunks = []
        current_chunk = ""
        for line in self.split_text_into_lines(text):
            line = line.strip()
            if not line:
                continue
            if len(current_chunk) + len(line) > max_length_split and current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            current_chunk = (current_chunk + " " + line).strip()
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def _build_table_aware_chunks(self, text, max_length_split):
        chunks = []
        current_chunk = ""
        for is_table, block in self._split_text_blocks_for_tables(text):
            block_chunks = (
                self._split_table_block(block, max_length_split)
                if is_table
                else self._split_long_text(block, max_length_split)
            )
            for block_chunk in block_chunks:
                if not block_chunk.strip():
                    continue
                if len(current_chunk) + len(block_chunk) + 2 > max_length_split:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = ""
                if is_table:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                    chunks.append(block_chunk.strip())
                else:
                    current_chunk = (current_chunk + "\n\n" + block_chunk).strip()
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        return chunks

    def get_relationships_from_llm(
        self, model, text, terms, max_length_split=2000, **kwargs
    ):
        """
        Identify semantic relationships between terms.

        Args:
            model (str): Model to use.
            text (str): Full source text.
            terms (list): List of known terms to relate.
            max_length_split (int): Chunk size.
            **kwargs: Extra arguments for LLM.

        Returns:
            list: Triplets (term1, relation, term2).
        """
        relationships = []
        set_terms = set(terms)

        chunks = self._build_chunks(text, max_length_split)

        for chunk in chunks:
            if len(chunk.strip()) == 0:
                continue
            prompt = self._relationship_prompt().format(
                CONTEXT=chunk, VOCABULARY="\n".join(set_terms)
            )
            response = self.query_fn(model, prompt, **kwargs)

            for l in response.split("\n"):
                if len(l) > 0 and l[0] == "-" and len(l[1:].strip()) > 0:
                    l = l[1:].strip()
                    split = l.split(">")
                    if len(split) != 3:
                        continue
                    t1, rel, t2 = (
                        split[0].strip(),
                        split[1].strip(),
                        split[2].strip(),
                    )
                    t1, rel, t2 = (
                        t1.replace("*", "").strip(),
                        rel.replace("*", "").strip(),
                        t2.replace("*", "").strip(),
                    )

                    if len(t1) == 0 or len(rel) == 0 or len(t2) == 0:
                        continue
                    relationships.append((t1, rel, t2))
        return relationships

    def get_definitions_from_llm(
        self, model, text, terms, max_length_split=2000, **kwargs
    ):
        """
        Retrieve definitions for known terms.

        Args:
            model (str): LLM model name.
            text (str): Full source text.
            terms (list): Terms to define.
            max_length_split (int): Max characters per prompt.
            **kwargs: Extra params for LLM.

        Returns:
            dict: Mapping of term -> definition.
        """
        definitions = {}
        set_terms = set(terms)

        chunks = self._build_chunks(text, max_length_split)
        for chunk in chunks:
            if len(chunk.strip()) == 0:
                continue
            prompt = prompt_definitions.format(
                CONTEXT=chunk, VOCABULARY="\n".join(set_terms)
            )
            response = self.query_fn(model, prompt, **kwargs)

            for l in response.split("\n"):
                if (
                    len(l) > 0
                    and l[0] == "-"
                    and len(l[1:].strip()) > 0
                    and "-" in l
                ):
                    l = l[1:].strip()  # remove initial '-'
                    split = l.split(":")
                    if len(split) != 2:
                        continue
                    term, defi = split[0].strip(), split[1].strip()
                    term, defi = (
                        term.replace("*", "").strip(),
                        defi.replace("*", "").strip(),
                    )

                    if len(term) == 0 or len(defi) == 0:
                        continue
                    definitions[term] = defi

        return definitions

    def get_acronyms_from_llm(
        self, model, text, terms, max_length_split=2000, **kwargs
    ):
        """
        Extract acronym definitions from the text.

        Args:
            model (str): Model name.
            text (str): Text to analyze.
            terms (list): Terms to match.
            max_length_split (int): Max length of each chunk.
            **kwargs: Additional parameters for the query.

        Returns:
            dict: Mapping of acronym -> full form.
        """
        acronyms = {}
        set_terms = set(terms)

        lines = self.split_text_into_lines(text)

        # build chunks of text with a maximum length of max_length_split
        chunks = []
        current_chunk = ""
        for line in lines:
            line = line.strip()
            if len(current_chunk) + len(line) > max_length_split:
                chunks.append(current_chunk)
                current_chunk = ""
            current_chunk += line
        if len(current_chunk) > 0:
            chunks.append(current_chunk)

        for chunk in chunks:
            if len(chunk.strip()) == 0:
                continue
            prompt = prompt_acronym.format(
                CONTEXT=chunk, VOCABULARY="\n".join(set_terms)
            )
            response = self.query_fn(model, prompt, **kwargs)
            acronyms = {}
            for l in response.split("\n"):
                if len(l) > 0:
                    split = l.split(":")
                    if len(split) != 2:
                        continue
                    acronym, term = split[0].strip(), split[1].strip()
                    acronym, term = (
                        acronym.replace("*", "").strip(),
                        term.replace("*", "").strip(),
                    )
                    if len(acronym) == 0 or len(term) == 0:
                        continue
                    acronyms[acronym] = term
        return acronyms

    def get_acronyms_from_llm_full_text(self, model, text, terms, **params):
        """
        Extract acronyms without chunking the text.
        """
        set_terms = set(terms)
        prompt = prompt_acronym.format(
            CONTEXT=text, VOCABULARY="\n".join(set_terms)
        )
        response = self.query_fn(model, prompt, **params)
        acronyms = {}

        for l in response.split("\n"):
            if len(l) > 0:
                split = l.split(":")
                if len(split) != 2:
                    continue
                acronym, term = split[0].strip(), split[1].strip()
                if len(acronym.strip()) == 0 and len(term.strip()) == 0:
                    continue
                acronym, term = (
                    acronym.replace("*", "").strip(),
                    term.replace("*", "").strip(),
                )
                acronyms[acronym] = term
        return acronyms

    def postprocess_acronyms(self, acronyms, text, terms):
        """
        Remove acronyms that are not in the text
        """
        result = {}
        for acronym, term in acronyms.items():
            if (
                term.lower() in text.lower()
                and acronym.lower() in text.lower()
            ):
                result[acronym] = term
            else:
                print(
                    f"Removing acronym '{acronym}':'{term}' because it is not in the text"
                )
        return result

    def compute_matches_without_spaces(self, term, text):
        """
        Removes spaces from both the input term and input text and tries to find matches.
        The positions returned are with respect to the original text.
        All the text is converted to lower case.
        It is assumed that no match can happen in a substring of a term.
        """
        lower_text_no_spaces = text.lower().replace(" ", "")
        lower_term_no_spaces = term.lower().replace(" ", "")
        matches = [
            (m.start(), m.end(), term)
            for m in re.finditer(
                re.escape(lower_term_no_spaces), lower_text_no_spaces
            )
        ]
        if len(matches) == 0:
            return []
        matches_with_spaces = []
        matches.sort(key=lambda x: x[0])
        index_matches = 0
        index_no_space = 0
        for index_spaces in range(len(text)):
            if index_no_space == matches[index_matches][0]:
                matches_with_spaces.append(
                    (index_spaces, index_spaces + len(term), term)
                )
                index_matches += 1
                if index_matches == len(matches):
                    break
            if text[index_spaces] != " ":
                index_no_space += 1
        return matches_with_spaces

    def get_filtered_list_from_llm(
        self,
        model,
        text,
        space_separator=True,
        max_length_split=2000,
        remove_hallucinated=True,
        **kwargs,
    ):
        """
        Extract and match LLM-suggested terms to original text.

        Args:
            model (str): Model name.
            text (str): Source text.
            space_separator (bool): Whether to require space-bounded matches.
            max_length_split (int): Max characters per LLM chunk.
            **kwargs: Extra arguments to query function.

        Returns:
            list: Matched terms as tuples (term, start, end, sentence_index).
        """
        terms = self.get_list_from_llm(model, text, max_length_split, **kwargs)

        sentences = self.split_text_into_lines(text)

        # remove hallucinated terms
        all_terms = []
        for term in terms:
            term_matches = []
            c = 0  # sentence length accumulator
            for i, sentence in enumerate(sentences):
                sentence = sentence.lower()
                matches_start_space = [
                    (term, c + m.start() + 1, c + m.end(), i)
                    for m in re.finditer(
                        re.escape(" " + term.lower()), sentence
                    )
                ]
                matches_end_space = [
                    (term, c + m.start(), c + m.end() - 1, i)
                    for m in re.finditer(
                        re.escape(term.lower() + " "), sentence
                    )
                ]
                matches_start_dot = [
                    (term, c + m.start() + 1, c + m.end(), i)
                    for m in re.finditer(
                        re.escape("." + term.lower()), sentence
                    )
                ]
                matches_end_dot = [
                    (term, c + m.start(), c + m.end() - 1, i)
                    for m in re.finditer(
                        re.escape(term.lower() + "."), sentence
                    )
                ]
                matches_start_comma = [
                    (term, c + m.start() + 1, c + m.end(), i)
                    for m in re.finditer(
                        re.escape("," + term.lower()), sentence
                    )
                ]
                matches_end_comma = [
                    (term, c + m.start(), c + m.end() - 1, i)
                    for m in re.finditer(
                        re.escape(term.lower() + ","), sentence
                    )
                ]
                matches_start_paren = [
                    (term, c + m.start() + 1, c + m.end(), i)
                    for m in re.finditer(
                        re.escape("(" + term.lower()), sentence
                    )
                ]
                matches_end_paren = [
                    (term, c + m.start(), c + m.end() - 1, i)
                    for m in re.finditer(
                        re.escape(term.lower() + ")"), sentence
                    )
                ]
                matches_no_space = [
                    (term, c + m.start(), c + m.end(), i)
                    for m in re.finditer(re.escape(term.lower()), sentence)
                ]
                # NOTE: we skip the no-space matching as it potentially creates too many false positives
                # matches_no_space_text = compute_matches_without_spaces(term, sentence)
                # matches = matches_start_space + matches_end_space + matches_no_space_text
                matches = (
                    matches_start_space
                    + matches_end_space
                    + matches_start_dot
                    + matches_end_dot
                    + matches_start_comma
                    + matches_end_comma
                    + matches_start_paren
                    + matches_end_paren
                )
                if not space_separator or self.context_kind != "generic":
                    matches += matches_no_space
                # remove duplicates from matches
                matches = self.remove_duplicated_terms(matches)
                term_matches += matches
                c += len(sentence) + 1  # +1 for the periods
            if len(term_matches) == 0:
                print(f"Term '{term}' not found in text")
                if not remove_hallucinated:
                    term_matches += [(term, -1, -1, -1)]
            all_terms += term_matches

        return all_terms
