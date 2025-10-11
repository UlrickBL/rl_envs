import verifiers as vf
from datasets import load_dataset, Dataset
from PIL import Image
import numpy as np
import random

SYSTEM_PROMPT = """You are a multimodal reranker."""

RERANK_PROMPT = """You are given a user query and several candidate documents (DOC_1, DOC_2, ...), each containing an image.
Rank the documents from most to least relevant to the query.
Return your ranked list exactly as a valid Python list like: [DOC_3, DOC_1, DOC_2, ...].
Do not add explanations, text, or commentary.
"""

class RerankerVLEnv(vf.SingleTurnEnv):
    """
    Environment for a Visual-Language reranker trained with GRPO.
    Dataset expected to have:
      - query (str)
      - ground_truth (list[int])
      - retrieved (list[int])
    And an image mapping or a nested structure to resolve image paths.
    """

    def format_dataset(
        self,
        dataset: Dataset,
        image_db: dict,
        system_prompt: str | None = None,
        max_docs: int = 10,
    ) -> Dataset:

        def format_prompt_fn(query, doc_tags):
            messages = []
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                })
            user_content = [{"type": "text", "text": f"User query: {query}\n\nDocuments:\n"}]
            for tag in doc_tags:
                user_content.append({"type": "text", "text": f"{tag}:"})
                user_content.append({"type": "image"})  # placeholder for image insertion
            user_content.append({"type": "text", "text": RERANK_PROMPT})
            messages.append({"role": "user", "content": user_content})
            return messages

        def preprocess_fn(example):
            retrieved = example["retrieved"][:max_docs]
            permuted = random.sample(retrieved, len(retrieved))
            doc_tags = [f"DOC_{i+1}" for i in range(len(permuted))]

            images = []
            for idx in permuted:
                img = Image.open(image_db[idx]).convert("RGB")
                img = img.resize((640, 640))
                images.append(img)

            messages = format_prompt_fn(example["query"], doc_tags)
            return {
                "prompt": messages,
                "answer": example["ground_truth"],
                "images": images,
                "doc_tags": doc_tags,
            }

        return dataset.map(preprocess_fn)

def reward_parseable_list(parser, completion, doc_tags):
    """
    Reward = 1 if model output is a valid list of provided DOC_X tags, else 0.
    """
    try:
        parsed = parser.parse_answer(completion)
        if isinstance(parsed, str):
            parsed = eval(parsed)
        if not isinstance(parsed, list):
            return 0.0
        parsed = [x.strip() for x in parsed]
    except Exception:
        return 0.0

    if set(parsed) != set(doc_tags):
        return 0.0
    for el in parsed:
        if el not in doc_tags:
            return 0.0
    return 1.0


def reward_ndcg(parser, completion, doc_tags, ground_truth):
    """
    Compute NDCG based on predicted ranking.
    ground_truth: list[int] representing indices of relevant documents.
    """
    try:
        pred = parser.parse_answer(completion)
        if isinstance(pred, str):
            pred = eval(pred)
        if not isinstance(pred, list):
            return 0.0
        pred = [x.strip() for x in pred if x in doc_tags]
    except Exception:
        return 0.0

    if not pred:
        return 0.0

    rel = []
    for doc in pred:
        rank_idx = doc_tags.index(doc)
        rel.append(1.0 if rank_idx in ground_truth else 0.0)

    def dcg(scores):
        return sum(s / np.log2(i + 2) for i, s in enumerate(scores))

    ideal = sorted(rel, reverse=True)
    ndcg = dcg(rel) / (dcg(ideal) + 1e-8)
    return float(ndcg)

def load_environment(
    dataset_name="your_reranker_dataset",
    image_db=None,
    num_examples=None,
    split="train",
    size=None,
    max_docs=10,
    **kwargs
) -> vf.Environment:
    """
    Loads a Visual Reranker Environment.
    - dataset: must have query, ground_truth, retrieved
    - image_db: dict[index] = path/to/image
    """
    dataset = load_dataset(dataset_name, split=split)
    if size:
        dataset = dataset.select(range(size))

    parser = vf.JSONListParser()  # expect a list like ["DOC_1", "DOC_2", ...]
    rubric = vf.Rubric(
        funcs=[reward_parseable_list, reward_ndcg],
        weights=[0.3, 0.7]
    )

    return RerankerVLEnv(
        dataset=dataset,
        system_prompt=SYSTEM_PROMPT,
        parser=parser,
        rubric=rubric,
        image_db=image_db,
        max_docs=max_docs,
        **kwargs
    )
