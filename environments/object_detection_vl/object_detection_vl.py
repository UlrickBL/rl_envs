import verifiers as vf
import json
from scipy.optimize import linear_sum_assignment
import verifiers as vf
from datasets import load_dataset
from datasets import Dataset
from PIL import Image

def compute_iou(boxA, boxB):
    """Compute IoU via Hungarian algorithm to match bounding box with different orders
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = boxA_area + boxB_area - inter
    return inter / union if union > 0 else 0

def parse_detection_list(output):
    """
    Parse something like:
    [
      {"bbox":[1,2,3,4], "label":"window"},
      ...
    ]
    """
    try:
        # Try direct JSON first
        return json.loads(output)
    except Exception:
        # fallback regex extraction
        import re
        pattern = r'\{\s*"bbox"\s*:\s*\[([0-9\.,\s]+)\]\s*,\s*"label"\s*:\s*"([^"]+)"\s*\}'
        matches = re.findall(pattern, output)
        results = []
        for bbox_str, label in matches:
            nums = [float(x) for x in bbox_str.split(",")]
            results.append({"bbox": nums, "label": label})
        return results if results else None

def reward_parseable(parser, completion, state, *args, **kwargs):
    pred = parse_detection_list(completion)
    if pred is None:
        return 0.0
    return 1.0

def reward_object_count(parser, completion, state, *args, **kwargs):
    pred = parse_detection_list(completion)
    if pred is None:
        return 0.0

    gt = state["info"]["gt"]
    return 1.0 / (1 + abs(len(pred) - len(gt)))

from scipy.optimize import linear_sum_assignment
import numpy as np

def reward_matching(parser, completion, state, *args, **kwargs):
    pred = parse_detection_list(completion)
    if pred is None:
        return 0.0

    gt = state["info"]["gt"]
    if len(pred) == 0 or len(gt) == 0:
        return 0.0

    n, m = len(pred), len(gt)
    cost = np.zeros((n, m))

    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            iou = compute_iou(p["bbox"], g["bbox"])
            label_correct = 1.0 if p["label"] == g["label"] else 0.0
            score = iou + 0.5 * label_correct
            cost[i, j] = -score

    row_ind, col_ind = linear_sum_assignment(cost)

    scores = []
    for r, c in zip(row_ind, col_ind):
        p = pred[r]
        g = gt[c]
        iou = compute_iou(p["bbox"], g["bbox"])
        label_ok = 1.0 if p["label"] == g["label"] else 0.0
        scores.append(0.7 * iou + 0.3 * label_ok)

    return float(np.mean(scores))


class BBoxEnv(vf.SingleTurnEnv):

    def __init__(self, dataset, *args, max_size=640, **kwargs):
        self.max_size = max_size
        super().__init__(dataset=dataset, *args, **kwargs)

    def format_dataset(self, dataset: Dataset, category="window"):
        def preprocess_fn(example):
            img = example["image"]
            if not isinstance(img, Image.Image):
                img = Image.open(img)
            img = img.convert("RGB")
            img = smart_resize(img, self.max_size)

            gt = example["coord"]  # bounding box list

            prompt = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": 'locate every instance that belongs to the following categories: {category}. For each window, report bbox coordinates, in JSON format like this: {"bbox_2d": [x1, y1, x2, y2], "label": "{category}""'},
                    {"type": "image"}
                ]
            }]

            return {
                "prompt": prompt,
                "answer": json.dumps(gt),  # not used directly; reward compares raw
                "images": [img],
                "info": {"gt": gt}
            }

        return dataset.map(preprocess_fn)


def load_bbox_env(dataset, split="train", **kwargs):
    parser = vf.Parser()
    
    
    dataset = load_dataset("ulrickBL/archi_bbox",split=split) # WIP for the dataset
    
    rubric = vf.Rubric(
        funcs=[
            reward_parseable,
            reward_object_count,
            reward_matching,
        ],
        weights=[0.2, 0.2, 0.6]
    )

    return BBoxEnv(
        dataset=dataset,
        parser=parser,
        rubric=rubric,
        **kwargs
    )
