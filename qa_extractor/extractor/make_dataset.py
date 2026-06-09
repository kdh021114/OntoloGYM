#coding=utf8
import json, os, sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from runtime import load_config


qa_config = load_config()


def make_dataset(example_dir: str | None = None, output_path: str | None = None):
    example_dir = os.fspath(example_dir or qa_config.EXAMPLE_DIR)
    output_path = os.fspath(output_path or qa_config.OUTPUT_DATASET_PATH)
    os.makedirs(example_dir, exist_ok=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    used_uuids = []
    for example in os.listdir(example_dir):
        if example.endswith(".json"):
            example = json.load(open(os.path.join(example_dir, example), "r", encoding="utf-8"))
            if "from" in example:
                for qid in example["from"]:
                    used_uuids.append(qid)
    
    examples = []
    for example in os.listdir(example_dir):
        if example.endswith(".json"):
            example = json.load(open(os.path.join(example_dir, example), "r", encoding="utf-8"))
            if example["uuid"] not in used_uuids:
                examples.append(example)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    print(f"Generated a dataset with {len(examples)} examples from {example_dir} into {output_path}.")

if __name__ == "__main__":
    make_dataset()
