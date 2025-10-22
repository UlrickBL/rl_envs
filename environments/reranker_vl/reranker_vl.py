import random
import re
import numpy as np
from PIL import Image
from datasets import Dataset, load_dataset
import verifiers as vf
import json

SYSTEM_PROMPT = """You are given a user query and several candidate documents (DOC_1, DOC_2, ...), each containing an image.
Rank the documents from most to least relevant to the query.
Return your ranked list exactly as a valid Python list like: [DOC_3, DOC_1, DOC_2, ...].
Do not add explanations, text, or commentary.
"""

RERANK_PROMPT = """Here are the query and documents :"""

def smart_resize(image: Image.Image, max_size=640):
    """Resize proportionally so the longest side equals max_size."""
    w, h = image.size
    scale = max_size / max(w, h)
    new_size = (int(w * scale), int(h * scale))
    return image.resize(new_size, Image.LANCZOS)

class RerankerVLEnv(vf.SingleTurnEnv):
    """
    Visual-Language Reranker Environment for query → images (1 positive + sampled negatives).
    """
    def __init__(self, *args, image_db=None, max_docs=10, max_size=640, **kwargs):
        # assign before calling super() so they're available early
        self.image_db = image_db
        self.max_docs = int(max_docs)
        self.max_size = int(max_size)

        print("init RerankerVLEnv")
        print("max_docs =", self.max_docs)
        print("image_db =", self.image_db)

        # now initialize parent
        super().__init__(*args, **kwargs)
        
    def format_dataset(self, dataset: Dataset, system_prompt: str | None = None, few_shot=None) -> Dataset:
        if isinstance(dataset, dict):
            dataset = dataset["train"]
    
        def format_prompt_fn(query, doc_tags):
            messages = []
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                })
    
            user_content = [{"type": "text", "text": f"User query: {query}\n\nDocuments:\n"}]
            for tag in doc_tags:
                user_content.append({"type": "text", "text": f"{tag}:"})
                user_content.append({"type": "image"})
            user_content.append({"type": "text", "text": RERANK_PROMPT})
    
            messages.append({"role": "user", "content": user_content})
            return messages
    
        def preprocess_fn(example):
            negs = example["neg_ids"]
            sampled_negs = random.sample(negs, min(len(negs), self.max_docs - 1))
            doc_ids = [example["pos_id"]] + sampled_negs
            random.shuffle(doc_ids)
            doc_tags = [f"DOC_{i+1}" for i in range(len(doc_ids))]
    
            images = []
            for idx in doc_ids:
                img = self.image_db["corpus"][idx]["image"]
                if not isinstance(img, Image.Image):
                    img = Image.open(img)
                img = img.convert("RGB")
                img = smart_resize(img, self.max_size)
                images.append(img)
    
            pos_idx = doc_ids.index(example["pos_id"])
            answer = doc_tags[pos_idx]
            messages = format_prompt_fn(example["query"], doc_tags)
    
            return {
                "prompt": messages,
                "answer": answer,
                "images": images,
                "info" : {"doc_tags": doc_tags,
                        "doc_ids": doc_ids,}
            }
    
        formatted = dataset.map(preprocess_fn)
        print("Columns in formatted dataset:", formatted.column_names)
        return formatted


def parse_doc_list_robust(completion):
    """
    Robustly parses a string completion into a list of document strings.
    Handles both quoted ('DOC_X') and unquoted (DOC_X) entries.
    """
    completion = completion.strip()
    if not (completion.startswith("[") and completion.endswith("]")):
        return None
    try:
        content = completion[completion.index('[') + 1: completion.rindex(']')]
    except ValueError:
        return None
    items = re.findall(r"[\w-]+", content) 
    parsed_list = [item for item in items if item]
    return parsed_list

def reward_mrr(parser, completion, answer):
    """
    Compute MRR (Mean Reciprocal Rank) for predicted ranking.
    - completion: model output (should be a Python list of DOC_X strings)
    - answer: the correct DOC_X label (e.g. "DOC_3")
    """
    try:
        pred = parser.parse_answer(completion)
        if isinstance(pred, str):
            pred = parse_doc_list_robust(pred)
        if not isinstance(pred, list):
            return 0.0
        pred = [x.strip() for x in pred]
    except Exception:
        return 0.0

    if not pred:
        return 0.0
    try:
        rank = pred.index(answer)
        return 1.0 / (rank + 1)
    except ValueError:
        return 0.0


def reward_parseable_list_only(parser, completion, *args, **kwargs):
    """
    Reward is 1 if output is exactly a parsable list and nothing else.
    Anything else (extra text, explanation, malformed) is 0.
    """
    try:
        pred = parser.parse_answer(completion)
        if isinstance(pred, str):
            pred_eval = parse_doc_list_robust(pred)
            if pred_eval == None :
                return 0
        else:
            pred_eval = pred
        completion_stripped = completion.strip()
        if not (completion_stripped.startswith("[") and completion_stripped.endswith("]")):
            return 0.0
        if not isinstance(pred_eval, list):
            return 0.0
        if not all(isinstance(x, str) for x in pred_eval):
            return 0.0
        return 1.0
    except Exception:
        return 0.0


def reward_valid_doc_tags(parser, completion, state, *args, **kwargs):
    """
    Reward = 1 if list contains exactly the same DOC_X tags (no duplicates, no missing),
    and the count matches the number of images.
    """
    try:
        pred = parser.parse_answer(completion)
        if isinstance(pred, str):
            pred = parse_doc_list_robust(pred)
        if not isinstance(pred, list):
            return 0.0
        pred = [x.strip() for x in pred]
    except Exception:
        return 0.0

    if set(pred) == set(state["info"]["doc_tags"]) and len(pred) == len(state["info"]["doc_tags"]):
        return 1.0
    else:
        return 0.0



def load_environment(
    dataset_name="UlrickBL/mm_reranker_rl_train",
    split="train",
    size=None,
    max_docs=10,
    max_size=640,
    **kwargs
) -> vf.Environment:
    dataset = load_dataset(dataset_name, "train")
    image_db = load_dataset(dataset_name, "corpus")
    if size:
        dataset = dataset.select(range(size))

    rubric = vf.Rubric(
        funcs=[
            reward_mrr,
            reward_parseable_list_only,
            reward_valid_doc_tags
        ],
        weights=[0.6, 0.2, 0.2]
    )

    parser = vf.Parser()
    
    return RerankerVLEnv(
        dataset=dataset,
        system_prompt=SYSTEM_PROMPT,
        parser=parser,
        rubric=rubric,
        image_db=image_db,
        max_docs=max_docs,
        max_size=max_size,
        **kwargs
    )
