#coding=utf8
import os, sys, json
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from runtime import load_config
from extractor.tracker_prompt import TRACK_PROMPT, EVALUATOR_PROMPT, USECASE_PROMPT
from utils.llm_utils import call_llm_with_pattern, call_llm, convert_to_message, call_llm_with_message

qa_config = load_config()

EVALUATIONS_FILE = Path(qa_config.EVALUATIONS_FILE)
EVALUATIONS = json.load(open(EVALUATIONS_FILE, 'r', encoding='utf-8'))
EVALUATORS_PROMPT = ""
for eval_func in EVALUATIONS:
    evaluation = EVALUATIONS[eval_func]
    EVALUATORS_PROMPT += EVALUATOR_PROMPT.format(
        function = eval_func,
        description = evaluation['description'],
        parameters = evaluation['parameters'],
        use_cases = "\n".join([
            USECASE_PROMPT.format(
                index = idx,
                example = usecase['example'],
                explanation = usecase['explanation']
            ) for idx, usecase in enumerate(evaluation['use_cases'], start=1)
        ])
    )


def build_eval_preference_instruction(eval_preference: str | None) -> str:
    eval_preference = str(eval_preference or "").strip().lower()
    if not eval_preference or str(getattr(qa_config, "EVAL_BALANCE_MODE", "soft")).lower() == "off":
        return ""
    if eval_preference not in {"objective", "subjective"}:
        return ""

    hints = dict(getattr(qa_config, "EVAL_PREFERENCE_HINTS", {}) or {})
    hint = hints.get(eval_preference, "")
    return (
        f"- Dataset balance preference: prefer a naturally `{eval_preference}` evaluator "
        "when the question and answer genuinely support it. Do not force the evaluator type; "
        "if the answer is better judged by the other type, choose the other type honestly. "
        f"{hint}"
    )


class BaseTracker(ABC):
    model: str = None
    temperature: float = None
    
    def __init__(self, model: str, temperature: float):
        self.model = model
        self.temperature = temperature
        
    @abstractmethod
    def _track(self, **kwargs):
        pass
    
    def track(self, **kwargs):
        return self._track(**kwargs)

class SingleTracker(BaseTracker):
    """ Moderate the question and fill the other parameters.
    1. Reform the question and the answer.
    2. Consider `evaluator`, `answer_format` and `tags`.
    TODO: Moderate human examples.
    """
    def __init__(self, model, temperature):
        super().__init__(model, temperature)
    
    def _track_with_llm(
            self,
            messages: List[Dict[str, Any]],
            template: str
        ) -> List[Any]:
        messages = str(messages)
        if len(messages) >= 50000:
            messages = messages[:50000]
        trajectory = {
            "role": "user", 
            "content": f"Here are original trajectory where the question and answer are generated:\n```json\n{messages}\n```"
        }
        messages = convert_to_message(template)
        messages.append(trajectory)
        response = call_llm_with_message(messages, model=self.model, temperature=self.temperature)
        messages.append({"role": "assistant", "content": response})
        return messages
    
    def _track(self, **kwargs) -> List[Any]:
        messages: List[Dict[str, Any]] = kwargs.get('messages', [])
        question: str = kwargs.get('question', '')
        answer: str = kwargs.get('answer', '')
        eval_preference = kwargs.get('eval_preference', '')
        template = TRACK_PROMPT.format(
            evaluator = EVALUATORS_PROMPT,
            eval_preference_instruction = build_eval_preference_instruction(eval_preference),
            question = question,
            answer = answer
        )
        return self._track_with_llm(messages, template)

class MultipleTracker(BaseTracker):
    pass

class RetrievalTracker(BaseTracker):
    pass

class ComprehensiveTracker(SingleTracker):
    pass
