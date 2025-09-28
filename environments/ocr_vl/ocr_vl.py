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

class OCRVLEnv(vf.SingleTurnEnv):
    """
    Environment for OCR with VLM
    """
    def format_dataset(
            self,
            dataset: Dataset,
            system_prompt: str | None = None,
            answer_key: str = "transcription",
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
                "answer": example["word"],
                "image": example["image"],
            }
        
        return dataset.map(preprocess_fn)

def reward_answer(parser, completion, answer):
    """
    Reward OCR task based on :
    - CER when majority is numbers or punctuation
    - WER else
    """
    response = parser.parse_answer(completion) or ''

    num_punct_chars = sum(c.isdigit() or re.match(r"\W", c) for c in answer)
    ratio = num_punct_chars / max(1, len(answer))

    if ratio > 0.5:  
        score = cer_metric.compute(predictions=[response], references=[answer])
    else:
        score = wer_metric.compute(predictions=[response], references=[answer])
    reward = 1 - score  
    return reward

def load_environment(num_examples=None,split="train[:1000]", **kwargs) -> vf.Environment:
    '''
    Loads OCR VL Environment.
    '''
    dataset = load_dataset("deepcopy/webui_multilingual_ocr",split=split)
    parser = vf.XMLParser(fields=["ocr"], answer_field="ocr")
    rubric = vf.Rubric(
        funcs=[reward_answer, parser.get_format_reward_func()],
        weights=[1.0, 0.2]
    )

    return OCRVLEnv(
        dataset=dataset,
        system_prompt=SYSTEM_PROMPT,
        parser=parser,
        rubric=rubric,
        **kwargs
    )