import random
import numpy as np
from PIL import Image
from datasets import Dataset, load_dataset
import verifiers as vf

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
    def format_dataset(
        self,
        dataset: Dataset,
        max_size,
        image_db,
        system_prompt: str | None = None,
        images_number: int = 5,
    ) -> Dataset:

        def format_prompt_fn(query, doc_tags):
            """Creates the conversation message structure."""
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
            sampled_negs = random.sample(negs, min(len(negs), images_number - 1))

            doc_ids = [example["pos_id"]] + sampled_negs
            random.shuffle(doc_ids)

            doc_tags = [f"DOC_{i+1}" for i in range(len(doc_ids))]

            images = []
            for idx in doc_ids:
                img = Image.open(image_db[idx]).convert("RGB")
                img = smart_resize(img,max_size)
                images.append(img)

            pos_idx = doc_ids.index(example["pos_id"])
            answer = doc_tags[pos_idx]
            messages = format_prompt_fn(example["query"], doc_tags)

            return {
                "prompt": messages,
                "answer": answer,
                "images": images,
                "doc_tags": doc_tags,
                "doc_ids": doc_ids,
            }

        return dataset.map(preprocess_fn)

def reward_ndcg(parser, completion, doc_tags, ground_truth_tag):
    """
    Compute NDCG (Normalized Discounted Cumulative Gain) for predicted ranking.
    - completion: model output (should be a Python list of DOC_X strings)
    - ground_truth_tag: the correct DOC_X label (e.g. "DOC_3")
    """
    try:
        pred = completion
        if isinstance(pred, str):
            pred = eval(pred)
        if not isinstance(pred, list):
            return 0.0
        pred = [x.strip() for x in pred if x in doc_tags]
    except Exception:
        return 0.0
    if not pred:
        return 0.0
    rel = [1.0 if doc == ground_truth_tag else 0.0 for doc in pred]
    def dcg(scores):
        return sum(s / np.log2(i + 2) for i, s in enumerate(scores))
    ideal = sorted(rel, reverse=True)
    ndcg = dcg(rel) / (dcg(ideal) + 1e-8)
    return float(ndcg)


def reward_parseable_list_only(parser, completion, *args, **kwargs):
    """
    Reward = 1 if output is exactly a parsable list and nothing else.
    E.g. "[DOC_1, DOC_3, DOC_2]" → ✅
    Anything else (extra text, explanation, malformed) → 0.
    """
    try:
        pred = completion
        if isinstance(pred, str):
            pred_eval = eval(pred)
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


def reward_valid_doc_tags(parser, completion, doc_tags, *args, **kwargs):
    """
    Reward = 1 if list contains exactly the same DOC_X tags (no duplicates, no missing),
    and the count matches the number of images.
    """
    try:
        pred = completion
        if isinstance(pred, str):
            pred = eval(pred)
        if not isinstance(pred, list):
            return 0.0
        pred = [x.strip() for x in pred]
    except Exception:
        return 0.0

    if set(pred) == set(doc_tags) and len(pred) == len(doc_tags):
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
    dataset = load_dataset(dataset_name, split=split)
    image_db = load_dataset(dataset_name, split="corpus")
    if size:
        dataset = dataset.select(range(size))

    rubric = vf.Rubric(
        funcs=[
            reward_ndcg,
            reward_parseable_list_only,
            reward_valid_doc_tags
        ],
        weights=[0.6, 0.2, 0.2]
    )

    return RerankerVLEnv(
        dataset=dataset,
        system_prompt=SYSTEM_PROMPT,
        rubric=rubric,
        image_db=image_db,
        max_docs=max_docs,
        max_size=max_size,
        **kwargs
    )
