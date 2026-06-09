
#coding=utf8
import logging, re, os, sys, time, openai, base64, tiktoken
from pathlib import Path
from tiktoken import Encoding
from typing import List, Dict, Any, Optional, Union
from openai.types.chat.chat_completion import ChatCompletion

QA_ROOT = Path(__file__).resolve().parents[1]
if str(QA_ROOT) not in sys.path:
    sys.path.insert(0, str(QA_ROOT))

from runtime import load_config
from common.usage_logging import log_openai_usage


qa_config = load_config()

DEFAULT_LLM_MODEL = os.getenv('DEFAULT_LLM_MODEL') or qa_config.DEFAULT_LLM_MODEL
DEFAULT_TOP_P = float(os.getenv('DEFAULT_TOP_P') or qa_config.DEFAULT_TOP_P)
DEFAULT_TEMPERATURE = float(os.getenv('DEFAULT_TEMPERATURE') or qa_config.DEFAULT_TEMPERATURE)

ENCODING_MODELS = dict()
logger = logging.getLogger(__name__)


def _get_encoding_or_none(encoding_model: str = 'cl100k_base') -> Optional[Encoding]:
    if encoding_model in ENCODING_MODELS:
        return ENCODING_MODELS[encoding_model]
    try:
        encoding: Encoding = tiktoken.get_encoding(encoding_model)
    except Exception:
        return None
    ENCODING_MODELS[encoding_model] = encoding
    return encoding


def calculate_tokens(text: str, encoding_model: str = 'cl100k_base') -> int:
    """ Calculate the number of tokens in the text using the encoding_model tokenizer.
    """
    encoding = _get_encoding_or_none(encoding_model)
    if encoding is None:
        return max(1, len(text) // 4)
    tokens = encoding.encode(text)
    return len(tokens)


def truncate_tokens(text: str, max_tokens: int = 30, encoding_model: str = 'cl100k_base') -> str:
    """ Given a text string, truncate it to max_tokens * 1000 using encoding_model tokenizer
    """
    encoding = _get_encoding_or_none(encoding_model)
    if encoding is None:
        max_chars = max_tokens * 4000
        return text if len(text) <= max_chars else text[:max_chars]
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens * 1000:
        tokens = tokens[:max_tokens * 1000]
        text = encoding.decode(tokens)
    return text

def call_llm_with_message(
    messages: Any, 
    model: str = DEFAULT_LLM_MODEL, 
    top_p: float = DEFAULT_TOP_P, 
    temperature: float = DEFAULT_TEMPERATURE
) -> str:
    """ Call LLM to generate the response directly using the message list.
    """
    api_key = os.getenv('OPENAI_API_KEY', None)
    base_url = os.getenv('OPENAI_BASE_URL', None)
    last_error = None
    for attempt in range(1, 5):
        try:
            client = openai.OpenAI(api_key=api_key, base_url=base_url)
            completion: ChatCompletion = client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=temperature,
                top_p=top_p
            )
            log_openai_usage(completion, component="qa_extractor")
            return completion.choices[0].message.content.strip()
        except Exception as exc:
            last_error = exc
            if attempt == 4:
                break
            wait_seconds = 3 * attempt
            logger.warning("AirQA LLM call failed on attempt %s/4: %s", attempt, exc)
            time.sleep(wait_seconds)
    raise last_error

def convert_to_message(
        template: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
    """ Convert the template to the message list. The `template` merely supports the following format:
    {{system_message}}

    {{user_message}}
    Note that, the system and user messages should be separated by two consecutive newlines. And the first block is the system message, the other blocks are the user message. There is no assistant message or interaction history.
    
    If you need to add an image, you can set `image` in kwargs to the image message dict. See `get_image_message` in `utils.functions.image_functions` for the image message format.
    """
    system_msg = template.split('\n\n')[0].strip()
    user_msg = '\n\n'.join(template.split('\n\n')[1:]).strip()
    messages = [
        {
            "role": 'system',
            "content": system_msg
        }
    ]
    if user_msg:
        messages.append(
            {
                "role": 'user',
                "content": user_msg
            }
        )
    if kwargs.get("image", None):
        image_message = kwargs["image"]
        if type(image_message) is list:
            messages.extend(image_message)
        else:
            messages.append(image_message)
    return messages

def call_llm(
        template: str, 
        model: str = DEFAULT_LLM_MODEL, 
        top_p: float = DEFAULT_TOP_P, 
        temperature: float = DEFAULT_TEMPERATURE,
        **kwargs
    ) -> str:
    """ Automatically construct the message list from template and call LLM to generate the response. 
    See `convert_to_message` for the template format.
    """
    return call_llm_with_message(messages=convert_to_message(template, **kwargs), model=model, top_p=top_p, temperature=temperature)

def call_llm_with_pattern(
        template: str,
        pattern: str, 
        model: str = DEFAULT_LLM_MODEL,
        top_p: float = DEFAULT_TOP_P, 
        temperature: float = DEFAULT_TEMPERATURE,
        **kwargs
    ) -> List[str]:
    """ Automatically construct the message list from template, call LLM to generate the response, and parse the response with givern pattern.
    """
    response = call_llm(template=template, model=model, top_p=top_p, temperature=temperature, **kwargs)
    matched = re.findall(pattern, response, re.DOTALL)
    if len(matched) == 0:
        return None
    return [s.strip() for s in matched[-1]]

def get_image_mime_type(image_path: str) -> str:
    """ Get the mime type of the image file according to its extension.
    """
    ext = os.path.basename(image_path).split('.')[-1].lower()
    mime_types = {
        'bmp': 'image/bmp',
        'dib': 'image/bmp',
        'icns': 'image/icns',
        'ico': 'image/x-icon',
        'jfif': 'image/jpeg',
        'jpe': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'jpg': 'image/jpeg',
        'j2c': 'image/jp2',
        'j2k': 'image/jp2',
        'jp2': 'image/jp2',
        'jpc': 'image/jp2',
        'jpf': 'image/jp2',
        'jpx': 'image/jp2',
        'apng': 'image/png',
        'png': 'image/png',
        'bw': 'image/sgi',
        'rgb': 'image/sgi',
        'rgba': 'image/sgi',
        'sgi': 'image/sgi',
        'tif': 'image/tiff',
        'tiff': 'image/tiff',
        'webp': 'image/webp',
    }
    return mime_types.get(ext, 'image/jpeg')

def get_image_message(
        template: str,
        image_path: Optional[Union[List[str], str]] = None,
        base64_image: Optional[str] = None,
        mine_type: str = 'image/jpeg',
        image_limit: int = -1
    ) -> Dict[str, Any]:
    """ Get the image message for LLM calling.
    @args:
        template: str, the description/instruction for the image.
        image_path: str or List[str], path(s) to the image file(s) you want to summary (overwrite `base64_image`).
        base64_image: str, base64 encoded image string. Either `image_path` or `base64_image` must be provided.
        mine_type: str, the mine type of the image, should be specified if only `base64_image` is provided, default to 'image/jpeg'.
        image_limit: int, the maximum number of images to use, default to -1 (no limit).
    @return:
        message: dict, a role-content message pair
    """
    assert image_path is not None or base64_image is not None, "Either `image_path` or `base64_image` must be provided."
    message = {
        "role": "user",
        "content": [
            {
                'type': 'text',
                'text': template
            }
        ]
    }
    if image_path is not None:
        if not isinstance(image_path, list): # multiple images
            image_path = [image_path]
        for idx, img_path in enumerate(image_path):
            if image_limit > 0 and idx >= image_limit:
                print(f'[Warning]: exceeding the image count limit {image_limit}, only the first {image_limit} images will be used.')
                break
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image file {img_path} does not exist.")
            mine_type = get_image_mime_type(img_path)
            with open(img_path, 'rb') as f:
                base64_image = base64.b64encode(f.read()).decode('utf-8')
            message['content'].append({
                'type': 'image_url',
                "image_url": {
                    "url":  f"data:{mine_type};base64,{base64_image}"
                }
            })
    else:
        message['content'].append({
            'type': 'image_url',
            "image_url": {
                "url":  f"data:{mine_type};base64,{base64_image}"
            }
        })
    
    return message
