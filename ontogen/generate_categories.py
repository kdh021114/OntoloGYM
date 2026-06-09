"""
A CLI module to generate higher level categories from a collection of scientific papers.
"""
import argparse
import json
import logging
import os
import random
from pathlib import Path

try:
    import ollama
except ImportError:  # pragma: no cover - optional runtime dependency.
    ollama = None
try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional runtime dependency.
    OpenAI = None
from common.usage_logging import log_openai_usage
from utils import read_text, write_text

logging.basicConfig(level=logging.INFO)


prompt_generate_cats = """
I will provide you with texts from a collection of scientific papers of '{TOPIC}'. 
Your task is to analyze the text and identify the most relevant and general terms that define the key aspects of '{TOPIC}'. 
You should categorize these terms into main categories and their respective subcategories.

Please follow these steps:

    Read the abstracts and introductions carefully.
    Identify the most significant terms that are frequently mentioned and are central to '{TOPIC}'.
    Organize these terms into main categories, with each main category containing a few subcategories.
    Ensure that the categories and subcategories are simple and brief.

The output should be structured as follows:

Main Category 1: <Category name>

    Subcategory 1: <Subcategory name>
    Subcategory 2: <Subcategory name>

Main Category 2: <Category name>

    Subcategory 1: <Subcategory name>
    Subcategory 2: <Subcategory name>

Continue this pattern as needed to cover the primary aspects of '{TOPIC}'.
Do not conjunctions in category names, e.g., "<Category 1> and <Category 2>" should be two separate categories: "<Category 1>" and "<Category 2>".

Texts:
{CONTEXT}
"""

prompt_generate_evidence_cats = """
I will provide you with evidence-aware texts from a collection of scientific papers about '{TOPIC}'.
The texts may include provenance labels, paper sections, and serialized PDF tables.
Your task is to identify high-level ontology seed categories and subcategories that describe the recurring scientific concepts and evidence types in '{TOPIC}'.

Consider these high-level category families when they are supported by the texts:
Method, Dataset, Task, Metric, Experimental Setting, Hyperparameter, Result, Ablation Experiment, Component, Material / Entity, and Table-derived Evidence.

Use only information explicitly present in the provided texts. Numeric values and table result values should not become ontology categories; they are evidence relation objects for KG extraction.
Keep category and subcategory names simple and brief.

The output should be structured as follows:

Main Category 1: <Category name>

    Subcategory 1: <Subcategory name>
    Subcategory 2: <Subcategory name>

Main Category 2: <Category name>

    Subcategory 1: <Subcategory name>
    Subcategory 2: <Subcategory name>

Do not use conjunctions in category names, e.g., "<Category 1> and <Category 2>" should be two separate categories: "<Category 1>" and "<Category 2>".

Texts:
{CONTEXT}
"""

prompt_correct_format_cats = """
Given the following categories, format the categories and subcategories following the syntax below:

Main Category 1: <Category name>

    Subcategory 1: <Subcategory name>
    Subcategory 2: <Subcategory name>

Main Category 2: <Category name>

    Subcategory 1: <Subcategory name>
    Subcategory 2: <Subcategory name>
    Subcategory 3: <Subcategory name>

Main Category 3: <Category name>

    Subcategory 1: <Subcategory name>

Continue this pattern as needed to cover all the categories.
If a subcategory has subcategories, do not include them in the list. This is, include only the first and second level of the tree.

Here are the categories and subcategories:
{CATEGORIES}
"""

prompt_synthesize_cats = """
Given all these trees of categories, build a combined tree that captures the recurring categories and topics from the provided data, ensuring that only the most frequently mentioned categories are included.
In other words, create a more organized and concise list.
Do not include conjunctions in category names, e.g., "<Category 1> and <Category 2>" should be two separate categories: "<Category 1>" and "<Category 2>". 
Here is the list, separated by '-':

{CATEGORIES}

Remember your goal: build a combined tree that captures the recurring categories and topics from the provided data, ensuring that only the most frequently mentioned categories are included.
In other words, create a more organized and concise list.
Do not include conjunctions in category names, e.g., "<Category 1> and <Category 2>" should be two separate categories: "<Category 1>" and "<Category 2>". 
"""

prompt_curate_cats = """
Given the following ontology seed categories and subcategories, curate them into exactly {TARGET_COUNT} main categories.
This replaces the manual category selection step used in OntoGen: keep broad, reusable ontology categories and merge redundant or overly specific categories.

Rules:
- Use only categories supported by the provided list.
- If present, preserve these main categories because downstream KG/evaluation relies on them: {PROTECTED_CATEGORIES}.
- Preserve concise subcategories under each selected main category.
- Prefer categories that help organize scientific knowledge graphs across papers.
- Do not use numeric values, result values, paper-specific phrases, or conjunction-based category names as main categories.
- Return exactly {TARGET_COUNT} main categories unless fewer non-overlapping categories are present.

Use this exact format:

Main Category 1: <Category name>

    Subcategory 1: <Subcategory name>
    Subcategory 2: <Subcategory name>

Main Category 2: <Category name>

    Subcategory 1: <Subcategory name>

Categories to curate:
{CATEGORIES}
"""


prompt_self_reflect_fix_format_cats = """
Can you stop any error in the previous formatting? Recall that the format should be as follows:
---
Main Category 1: <Category name goes here>

    Subcategory 1: <Subcategory name goes here>
    Subcategory 2: <Subcategory name goes here>

Main Category 2: <Category name goes here>

    Subcategory 1: <Subcategory name goes here>
---
This is, the word "Main Category" should be followed by a number and a colon, and then the name of the category. The word "Subcategory" should be followed by a number, a colon, and then the name of the subcategory.
If you find any mistakes, please rewrite the whole list following the correct format. If you don't find any mistakes, respond with "The previous formatting is correct."
"""


def _openai_options(options):
    options = dict(options or {})
    if "max_tokens" in options and "max_completion_tokens" not in options:
        options["max_completion_tokens"] = options.pop("max_tokens")
    allowed = {
        "temperature",
        "top_p",
        "max_completion_tokens",
        "reasoning_effort",
        "verbosity",
        "seed",
    }
    return {key: value for key, value in options.items() if key in allowed and value is not None}


def query_chat(model, prompt, options, backend="ollama", base_url=None):
    if backend == "ollama":
        if ollama is None:
            raise ImportError("ollama is required for category generation with backend='ollama'.")
        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            options=options,
        )
        return response["message"]["content"]
    if backend in {"openai", "oai"}:
        if OpenAI is None:
            raise ImportError("openai is required for category generation with backend='openai'.")
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        )
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **_openai_options(options),
        )
        log_openai_usage(completion, component="ontogen_categories")
        return completion.choices[0].message.content or ""
    raise ValueError(f"Unknown category backend: {backend}")


def llm_format_cats(
    cats,
    model,
    options,
    backend="ollama",
    base_url=None,
):
    """
    This function receives a string with categories and subcategories which might be incorrectly formatted.
    The function uses an LLM to format the list of categories and subcategories to the same format,
    which is the following:

    Main Category 1: <Category name>

            Subcategory 1: <Subcategory name>
            Subcategory 2: <Subcategory name>

    Main Category 2: <Category name>

            Subcategory 1: <Subcategory name>


    The function will keep asking the user to fix the format until the format is correct or the maximum number of retries is reached.
    If the output of the LLM is incorrectly formated, a self reflection prompt is used to ask the model again to fix the format.

    Parameters:
    cats (str): String with categories and subcategories.
    model (str): Ollama model tag to use.
    options (dict): Dictionary with options to use in the model.
    """
    formated_prompt = prompt_correct_format_cats.format(CATEGORIES=cats)
    result = query_chat(model, formated_prompt, options, backend=backend, base_url=base_url)

    max_retry = 3
    while "Main Category" not in result or "    Subcategory" not in result:
        if "previous formatting is correct" in result.lower():
            fallback = loose_categories_to_str(cats)
            if fallback:
                return fallback
        retry_prompt = (
            formated_prompt
            + "\n\nPrevious answer:\n"
            + result
            + "\n\n"
            + prompt_self_reflect_fix_format_cats
        )
        result = query_chat(model, retry_prompt, options, backend=backend, base_url=base_url)
        max_retry -= 1
        if max_retry == 0:
            break
    if "Main Category" not in result or "    Subcategory" not in result:
        fallback = loose_categories_to_str(cats)
        if fallback:
            return fallback
    return result


def llm_synthesize_cats(cats, model, options, backend="ollama", base_url=None):
    """
    This function receives a string with categories and subcategories and asks an LLM to
    synthesize the categories and subcategories into a more concise list with the most frequently mentioned categories.
    If the output of the LLM is incorrectly formated, the model is prompted again with the same question. This is
    done to avoid answers where the output is empty or non-sensical.

    Parameters:
    cats (str): String with categories and subcategories.
    model (str): Ollama model tag to use.
    options (dict): Dictionary with options to use in the model.
    """
    formated_prompt = prompt_synthesize_cats.format(CATEGORIES=cats)
    result = query_chat(model, formated_prompt, options, backend=backend, base_url=base_url)
    if "Main Category" not in result or "    Subcategory" not in result:
        result = query_chat(model, formated_prompt, options, backend=backend, base_url=base_url)
    return result


def llm_curate_cats(
    cats,
    target_count,
    model,
    options,
    backend="ollama",
    base_url=None,
    protected_categories=None,
):
    """Ask an LLM to mimic OntoGen's manual category pruning step."""
    formated_prompt = prompt_curate_cats.format(
        CATEGORIES=cats,
        TARGET_COUNT=target_count,
        PROTECTED_CATEGORIES=", ".join(protected_categories or []) or "(none)",
    )
    result = query_chat(model, formated_prompt, options, backend=backend, base_url=base_url)
    if "Main Category" not in result:
        result = query_chat(model, formated_prompt, options, backend=backend, base_url=base_url)
    return result


def parse_cats_and_subcats(text):
    """
    This function receives a string with categories and subcategories with the following format:

    Main Category 1: <Category name>

            Subcategory 1: <Subcategory name>
            Subcategory 2: <Subcategory name>

    Main Category 2: <Category name>

            Subcategory 1: <Subcategory name>

    and returns a dictionary with the categories and subcategories.

    Parameters:
    text (str): String with categories and subcategories.
    """
    cats = {}
    current_cat = None
    for raw_line in text.splitlines():
        line = raw_line.strip().replace("*", "")
        if not line:
            continue
        if line.startswith("Main Category") and ":" in line:
            current_cat = line.split(":", 1)[1].strip()
            if current_cat:
                cats.setdefault(current_cat, [])
            continue
        if current_cat and line.startswith("Subcategory") and ":" in line:
            subcat = line.split(":", 1)[1].strip()
            if subcat:
                cats[current_cat].append(subcat)
    return cats


def parse_loose_cats_and_subcats(text):
    cats = {}
    current_cat = None
    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        indent = len(raw_line) - len(raw_line.lstrip())
        line = raw_line.strip()
        line = line.lstrip("-").strip().replace("**", "").replace("*", "").strip()
        if not line:
            continue
        if line.startswith("Main Category") and ":" in line:
            current_cat = line.split(":", 1)[1].strip()
            cats.setdefault(current_cat, [])
            continue
        if line.startswith("Subcategory") and ":" in line and current_cat:
            cats[current_cat].append(line.split(":", 1)[1].strip())
            continue
        if indent == 0:
            current_cat = line
            cats.setdefault(current_cat, [])
            continue
        if current_cat:
            cats[current_cat].append(line)
    return {cat: subs for cat, subs in cats.items() if cat and subs}


def strict_categories_to_str(cats):
    blocks = []
    for cat_index, (category, subcategories) in enumerate(cats.items(), start=1):
        lines = [f"Main Category {cat_index}: {category}", ""]
        for sub_index, subcategory in enumerate(subcategories, start=1):
            lines.append(f"    Subcategory {sub_index}: {subcategory}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def loose_categories_to_str(text):
    parsed = parse_cats_and_subcats(text)
    if not parsed:
        parsed = parse_loose_cats_and_subcats(text)
    return strict_categories_to_str(parsed) if parsed else ""


def enforce_protected_categories(candidate_text, curated_text, protected_categories, target_count):
    if not protected_categories:
        return curated_text
    candidates = parse_cats_and_subcats(candidate_text)
    curated = parse_cats_and_subcats(curated_text)
    if not candidates or not curated:
        return curated_text

    normalized_to_candidate = {category.lower(): category for category in candidates}
    for protected in protected_categories:
        candidate_name = normalized_to_candidate.get(str(protected).lower())
        if candidate_name and candidate_name not in curated:
            curated[candidate_name] = candidates[candidate_name]

    min_count = len([p for p in protected_categories if str(p).lower() in normalized_to_candidate])
    max_count = max(int(target_count or 0), min_count)
    if len(curated) > max_count:
        protected_lower = {str(name).lower() for name in protected_categories}
        pruned = {}
        for category, subcategories in curated.items():
            if category.lower() in protected_lower:
                pruned[category] = subcategories
        for category, subcategories in curated.items():
            if category.lower() in protected_lower:
                continue
            if len(pruned) >= max_count:
                break
            pruned[category] = subcategories
        curated = pruned
    return strict_categories_to_str(curated)


def cats_to_str(cats):
    """
    This function receives a dictionary with categories and subcategories and returns a string with the categories and subcategories,
    formatted as follows:

    <Main Category 1>
        <Subcategory 1>
        <Subcategory 2>
    --------------------
    <Main Category 2>
        <Subcategory 1>
        <Subcategory 2>
        <Subcategory 3>
    --------------------

    Parameters:
    cats (dict): Dictionary with categories and subcategories.
    """
    context = ""
    for answer in cats:
        if len(answer) == 0:
            continue
        for k in answer:
            k_clean = k.replace("*", "")
            context += k_clean + "\n"
            for sub in answer[k]:
                sub_clean = sub.replace("*", "")
                context += f"    {sub_clean}\n"
        context += "-" * 80 + "\n"
    return context


def count_main_categories(text):
    return len(parse_cats_and_subcats(text))


def resolve_curation_target_count(
    category_count,
    curation_ratio=None,
    curation_target_count=None,
    min_target_count=1,
):
    if category_count <= 0:
        return 0
    if curation_target_count is not None:
        target = int(curation_target_count)
    elif curation_ratio is not None:
        target = round(category_count * float(curation_ratio))
    else:
        target = category_count
    target = max(int(min_target_count or 1), target)
    return max(1, min(category_count, target))


def generate_categories(
    txt_files,
    main_topic,
    generation_model,
    format_model,
    synthesis_model,
    generation_backend="ollama",
    format_backend="ollama",
    synthesis_backend="ollama",
    generation_base_url=None,
    format_base_url=None,
    synthesis_base_url=None,
    output_dir="categories",
    evidence_aware=False,
    num_retries_consistency=20,
    num_generated_seed=20,
    max_chars_per_file=None,
    max_total_chars=None,
    generation_model_options=None,
    format_model_options=None,
    synthesis_model_options=None,
    run_llm_curation=False,
    curation_ratio=None,
    curation_target_count=None,
    curation_min_target_count=1,
    curation_protected_categories=None,
    curation_model=None,
    curation_backend=None,
    curation_base_url=None,
    curation_model_options=None,
):
    """
    Generate multiple possible categories seeds from a list of txt files.

    Parameters:
    txt_files (list): List of txt files to process.
    main_topic (str): Main topic to generate categories for.
    generation_model (str): Ollama model tag to use to generate categories.
    format_model (str): Ollama model tag to use to format categories.
    synthesis_model (str): Ollama model tag to use to synthesize frequent categories.
    num_retries_consistency (int): Number of retries to use with self-consistency.
            A larger number will increase the chances of getting consistent results. (default: 20)
    num_generated_seed (int): Number of seeds to generate. (default: 20)
    generation_model_options (dict): Dictionary with options to use in the generation model.
    format_model_options (dict): Dictionary with options to use in the format model.
    synthesis_model_options (dict): Dictionary with options to use in the synthesis model.
    """
    if generation_model_options is None:
        generation_model_options = {}
    if format_model_options is None:
        format_model_options = {}
    if synthesis_model_options is None:
        synthesis_model_options = {}
    if curation_model_options is None:
        curation_model_options = {}
    if curation_protected_categories is None:
        curation_protected_categories = []

    # random shuffle the list of files
    txt_files = [Path(txt_file) for txt_file in txt_files]
    random.shuffle(txt_files)

    context = ""
    for i, txt_file in enumerate(txt_files):
        text = read_text(txt_file)
        while "\n\n" in text:
            text = text.replace("\n\n", "\n")
        if max_chars_per_file is not None and len(text) > int(max_chars_per_file):
            text = text[: int(max_chars_per_file)].rstrip()
        block = f"Document {i} ({txt_file.name}): {text}\n\n"
        if max_total_chars is not None and len(context) + len(block) > int(max_total_chars):
            remaining = int(max_total_chars) - len(context)
            if remaining <= 0:
                break
            block = block[:remaining].rstrip() + "\n\n"
        context += block

    responses = []
    for retry in range(num_retries_consistency):
        logging.info(
            f"Generating categories Retry {retry + 1}/{num_retries_consistency}"
        )
        prompt_template = prompt_generate_evidence_cats if evidence_aware else prompt_generate_cats
        formated_prompt = prompt_template.format(
            CONTEXT=context, TOPIC=main_topic
        )
        response = query_chat(
            generation_model,
            formated_prompt,
            generation_model_options,
            backend=generation_backend,
            base_url=generation_base_url,
        )
        responses.append(response)
        logging.info(response)

    formated_responses = []
    responses = [c for c in responses if len(c.strip()) > 0]
    for i, response in enumerate(responses):
        logging.info(f"Formatting category {i + 1}/{len(responses)}")
        response = llm_format_cats(
            response,
            format_model,
            options=format_model_options,
            backend=format_backend,
            base_url=format_base_url,
        )
        formated_responses.append(response)
        logging.info(response)

    formated_responses_reduced = []
    for response in formated_responses:
        formated_responses_reduced.append(parse_cats_and_subcats(response))
    filtered_cats_text = cats_to_str(formated_responses_reduced)

    synthesized_cats = []
    for i in range(num_generated_seed):
        logging.info(f"Synthesizing categories {i + 1}/{num_generated_seed}")
        res = llm_synthesize_cats(
            filtered_cats_text,
            synthesis_model,
            options=synthesis_model_options,
            backend=synthesis_backend,
            base_url=synthesis_base_url,
        )
        synthesized_cats.append(res)
        logging.info(res)

    formated_responses = []
    responses = [c for c in responses if len(c.strip()) > 0]
    for i, response in enumerate(synthesized_cats):
        logging.info(f"Formatting category {i + 1}/{len(responses)}")
        response = llm_format_cats(
            response,
            format_model,
            options=format_model_options,
            backend=format_backend,
            base_url=format_base_url,
        )
        formated_responses.append(response)
        logging.info(response)

    categories_folder = Path(output_dir)
    categories_folder.mkdir(parents=True, exist_ok=True)
    if run_llm_curation:
        curated_responses = []
        curation_summary = []
        curation_model = curation_model or synthesis_model
        curation_backend = curation_backend or synthesis_backend
        for i, response in enumerate(formated_responses):
            candidate_count = count_main_categories(response)
            target_count = resolve_curation_target_count(
                candidate_count,
                curation_ratio=curation_ratio,
                curation_target_count=curation_target_count,
                min_target_count=curation_min_target_count,
            )
            raw_categories = parse_cats_and_subcats(response)
            protected_present_count = len(
                [
                    category
                    for category in raw_categories
                    if category.lower() in {str(name).lower() for name in curation_protected_categories}
                ]
            )
            target_count = max(target_count, protected_present_count)
            raw_path = categories_folder / f"{main_topic}_categories_seed.{i}.raw.txt"
            write_text(raw_path, response)
            if target_count and candidate_count > target_count:
                logging.info(
                    "Curating categories %s/%s from %s to %s main categories",
                    i + 1,
                    len(formated_responses),
                    candidate_count,
                    target_count,
                )
                curated = llm_curate_cats(
                    response,
                    target_count,
                    curation_model,
                    options=curation_model_options,
                    backend=curation_backend,
                    base_url=curation_base_url,
                    protected_categories=curation_protected_categories,
                )
                curated = llm_format_cats(
                    curated,
                    format_model,
                    options=format_model_options,
                    backend=format_backend,
                    base_url=format_base_url,
                )
                curated = enforce_protected_categories(
                    response,
                    curated,
                    curation_protected_categories,
                    target_count,
                )
            else:
                curated = response
            curated_count = count_main_categories(curated)
            curated_responses.append(curated)
            curation_summary.append(
                {
                    "seed_index": i,
                    "raw_path": str(raw_path),
                    "candidate_count": candidate_count,
                    "target_count": target_count,
                    "curated_count": curated_count,
                }
            )
        formated_responses = curated_responses
        write_text(
            categories_folder / f"{main_topic}_category_curation_summary.json",
            json.dumps(curation_summary, ensure_ascii=False, indent=2),
        )

    output_paths = []
    for i, res in enumerate(formated_responses):
        output_path = categories_folder / f"{main_topic}_categories_seed.{i}.txt"
        write_text(output_path, res)
        output_paths.append(output_path)
    return output_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate possible categories seeds from a list of txt files."
    )
    parser.add_argument(
        "main_topic", type=str, help="Main topic to generate categories for."
    )
    parser.add_argument(
        "txt_files",
        type=str,
        nargs="+",
        help="Path to the text files to process.",
    )
    parser.add_argument(
        "--generation-model",
        "-gm",
        help="Ollama model tag to use to generate categories.",
        type=str,
    )
    parser.add_argument(
        "--generation-temperature",
        "-gt",
        help="Model temperature to use to generate categories.",
        type=float,
    )
    parser.add_argument(
        "--generation-num-ctx",
        "-gc",
        help="Context length in tokens to use to generate categories.",
        type=int,
    )
    parser.add_argument(
        "--generation-backend",
        choices=["ollama", "openai", "oai"],
        default="ollama",
        help="LLM backend to use when generating categories.",
    )
    parser.add_argument(
        "--generation-base-url",
        default=None,
        help="Optional OpenAI-compatible base URL for category generation.",
    )
    parser.add_argument(
        "--format-model",
        "-fm",
        help="Ollama model tag to use to format categories.",
        type=str,
    )
    parser.add_argument(
        "--format-temperature",
        "-ft",
        help="Model temperature to use to format categories.",
        type=float,
    )
    parser.add_argument(
        "--format-num-ctx",
        "-fc",
        help="Context length in tokens to use to format categories.",
        type=int,
    )
    parser.add_argument(
        "--format-backend",
        choices=["ollama", "openai", "oai"],
        default="ollama",
        help="LLM backend to use when formatting categories.",
    )
    parser.add_argument(
        "--format-base-url",
        default=None,
        help="Optional OpenAI-compatible base URL for category formatting.",
    )
    parser.add_argument(
        "--synthesis-model",
        "-sm",
        help="Ollama model tag to use to synthesize categories.",
        type=str,
    )
    parser.add_argument(
        "--synthesis-temperature",
        "-st",
        help="Model temperature to use to synthesize categories.",
        type=float,
    )
    parser.add_argument(
        "--synthesis-num-ctx",
        "-sc",
        help="Context length in tokens to use to synthesize categories.",
        type=int,
    )
    parser.add_argument(
        "--synthesis-backend",
        choices=["ollama", "openai", "oai"],
        default="ollama",
        help="LLM backend to use when synthesizing categories.",
    )
    parser.add_argument(
        "--synthesis-base-url",
        default=None,
        help="Optional OpenAI-compatible base URL for category synthesis.",
    )
    parser.add_argument(
        "--num-retries",
        "-r",
        help="Number of retries to ensure consistency.",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--num-generated-seed",
        "-s",
        help="Number of generated seeds.",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--output-dir",
        help="Directory where category seed files are written.",
        default="categories",
    )
    parser.add_argument(
        "--evidence-aware",
        help="Use prompts that account for sections and PDF tables.",
        action="store_true",
    )

    args = parser.parse_args()
    main_topic = args.main_topic
    txt_files = args.txt_files
    num_retries = args.num_retries
    num_generated_seed = args.num_generated_seed

    generation_model = args.generation_model
    format_model = args.format_model
    synthesis_model = args.synthesis_model

    options_generation = {}
    if args.generation_temperature:
        options_generation["temperature"] = args.generation_temperature
    if args.generation_num_ctx:
        options_generation["num_ctx"] = args.generation_num_ctx

    options_format = {}
    if args.format_temperature:
        options_format["temperature"] = args.format_temperature
    if args.format_num_ctx:
        options_format["num_ctx"] = args.format_num_ctx

    options_synthesis = {}
    if args.synthesis_temperature:
        options_synthesis["temperature"] = args.synthesis_temperature
    if args.synthesis_num_ctx:
        options_synthesis["num_ctx"] = args.synthesis_num_ctx

    generate_categories(
        txt_files,
        main_topic,
        generation_model,
        format_model,
        synthesis_model,
        generation_backend=args.generation_backend,
        format_backend=args.format_backend,
        synthesis_backend=args.synthesis_backend,
        generation_base_url=args.generation_base_url,
        format_base_url=args.format_base_url,
        synthesis_base_url=args.synthesis_base_url,
        output_dir=args.output_dir,
        evidence_aware=args.evidence_aware,
        num_retries_consistency=num_retries,
        num_generated_seed=num_generated_seed,
        generation_model_options=options_generation,
        format_model_options=options_format,
        synthesis_model_options=options_synthesis,
    )
