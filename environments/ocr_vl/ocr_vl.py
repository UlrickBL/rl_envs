import verifiers as vf
from datasets import load_dataset
from datasets import Dataset
from verifiers.types import ChatMessage
from io import BytesIO
import base64
from PIL import Image
from evaluate import load
import re

cer_metric = load("cer")
wer_metric = load("wer")

SYSTEM_PROMPT = """You are an expert OCR transcriptor"""

OCR_PROMPT = """Carefully analyze this image and transcribe exactly all the text that appears in it. Do not correct mistakes, do not rephrase, and do not add anything: return only the text as it is written in the image, preserving capitalization, punctuation, line breaks, and formatting as closely as possible. Put the output inside <ocr>[transcription]<\ocr>"""

def has_non_roman_chars(text: str) -> bool:
    """
    Checks if a string contains characters from common non-Roman scripts
    like Devanagari (Hindi), CJK (Chinese/Japanese), or Kana (Japanese).
    """
    # Unicode ranges:
    # U+0900 to U+097F: Devanagari (Hindi)
    # U+4E00 to U+9FFF: CJK Unified Ideographs (Chinese, Japanese, Korean)
    # U+3040 to U+309F: Hiragana (Japanese)
    # U+30A0 to U+30FF: Katakana (Japanese)
    non_roman_pattern = re.compile(r'[\u0900-\u097F\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF]')
    return bool(non_roman_pattern.search(text))
    
class OCRVLEnv(vf.SingleTurnEnv):
    """
    Environment for OCR with VLM
    """
    def format_dataset(
            self,
            dataset: Dataset,
            system_prompt: str | None = None,
            answer_key: str = "text",
            image_key: str = "image"
        ) -> Dataset:
        
        def format_prompt_fn() -> list[dict]:
            messages = []
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                })
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": OCR_PROMPT},
                ],
            })
            return messages
        
        def preprocess_fn(example):
            return {
                "prompt": format_prompt_fn(),
                "answer": example["text"],
                "image": example["image"],
            }
        return dataset.map(preprocess_fn)

def reward_answer(parser, completion, answer):
    """
    Reward OCR task, ensuring the reward is always in the [0, 1] range.

    - Forces CER for non-Roman scripts (Hindi, Chinese, Japanese).
    - For Roman scripts, uses CER if the answer is mostly numbers/punctuation,
      and WER otherwise.
    """
    if not answer or not isinstance(answer, str):
        return 0.0

    response = parser.parse_answer(completion) or ''
    score = 0.0

    if has_non_roman_chars(answer):
        score = cer_metric.compute(predictions=[response], references=[answer])
    else:
        num_punct_chars = sum(1 for c in answer if c.isdigit() or not c.isalnum())
        ratio = num_punct_chars / len(answer)

        if ratio > 0.5:
            score = cer_metric.compute(predictions=[response], references=[answer])
        else:
            score = wer_metric.compute(predictions=[response], references=[answer])

    reward = max(0.0, 1.0 - score)
    
    return reward


def load_environment(num_examples=None,split="train",lang=None,size=1000, **kwargs) -> vf.Environment:
    '''
    Loads OCR VL Environment.
    '''
    dataset = load_dataset("deepcopy/webui_multilingual_ocr",split=split)
    if lang :
        dataset = dataset.filter(lambda x: x["lang"] == lang)
    if size :
        dataset = dataset.select(range(1000))
    print(dataset)
    parser = vf.XMLParser(fields=["ocr"], answer_field="ocr")
    rubric = vf.Rubric(
        funcs=[reward_answer, parser.get_format_reward_func()],
        weights=[0.5, 0.5]
    )

    return OCRVLEnv(
        dataset=dataset,
        system_prompt=SYSTEM_PROMPT,
        parser=parser,
        rubric=rubric,
        **kwargs
    )