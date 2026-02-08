import verifiers as vf
import json
from scipy.optimize import linear_sum_assignment
from datasets import load_dataset,Dataset
from PIL import Image
import numpy as np
import re

def smart_resize(image: Image.Image, max_size=640): 
    """Resize proportionally so the longest side equals max_size.""" 
    w, h = image.size 
    scale = max_size / max(w, h) 
    new_size = (int(w * scale), int(h * scale)) 
    return image.resize(new_size, Image.LANCZOS)
    
def compute_iou(boxA, boxB):
    """Compute IoU"""
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

def normalize_detections(dets):
    """
    Convert model outputs into a standard format:
    [{"bbox": [...], "label": "..."}]
    """
    if not isinstance(dets, list):
        return None

    normalized = []
    for d in dets:
        if not isinstance(d, dict):
            continue

        bbox = d.get("bbox") or d.get("bbox_2d")
        label = d.get("label")

        if bbox is None or label is None:
            continue

        if len(bbox) != 4:
            continue

        normalized.append({
            "bbox": [float(x) for x in bbox],
            "label": str(label)
        })

    return normalized if normalized else None

def parse_detection_list(output):
    """
    Parse model output containing a list of detections.
    Accepts bbox or bbox_2d.
    """
    try:
        data = json.loads(output)
        return normalize_detections(data)
    except Exception:
        pass

    pattern = r'\{\s*"bbox(?:_2d)?"\s*:\s*\[([0-9\.,\s]+)\]\s*,\s*"label"\s*:\s*"([^"]+)"\s*\}'
    matches = re.findall(pattern, output)

    results = []
    for bbox_str, label in matches:
        nums = [float(x) for x in bbox_str.split(",")]
        if len(nums) == 4:
            results.append({"bbox": nums, "label": label})

    return results if results else None


def parse_ground_truth(gt_string):
    return parse_detection_list(
        gt_string,
    )

def parse_prediction(completion, parser):
    text = parser.parse_answer(completion) or ""
    return parse_detection_list(
        text
    )


def reward_parseable(parser, completion, state, *args, **kwargs):
    pred = parse_prediction(completion,parser)
    if pred is None:
        return 0.0
    return 1.0

def reward_object_count(parser, completion, state, *args, **kwargs):
    pred = parse_prediction(completion,parser)
    if pred is None:
        return 0.0

    gt = parse_ground_truth(state["info"]["gt"])
    return 1.0 / (1 + abs(len(pred) - len(gt)))

def reward_matching(parser, completion, state, *args, **kwargs):
    pred = parse_prediction(completion,parser)
    if pred is None:
        return 0.0

    gt = parse_ground_truth(state["info"]["gt"])
    if len(pred) == 0 or len(gt) == 0:
        return 0.0

    n, m = len(pred), len(gt)
    cost = np.zeros((n, m))

    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            iou = compute_iou(p["bbox"], g["bbox"])
            label_correct = 1.0 if p["label"] == g["label"] else 0.0
            cost[i, j] = -(iou + 0.5 * label_correct)

    row_ind, col_ind = linear_sum_assignment(cost)

    scores = []
    for r, c in zip(row_ind, col_ind):
        p = pred[r]
        g = gt[c]
        iou = compute_iou(p["bbox"], g["bbox"])
        label_ok = 1.0 if p["label"] == g["label"] else 0.0
        scores.append(0.7 * iou + 0.3 * label_ok)

    return float(np.mean(scores))


def reward_geometry_smooth_hungarian(parser, completion, state, *args, **kwargs):
    """
    Calculates spatial accuracy using Hungarian matching and Smooth IoU (distance hints).
    """
    gt = parse_ground_truth(state["info"]["gt"],parser)
    if gt is None or len(gt) == 0:
        return 0.0

    pred = parse_prediction(completion,parser)
    if pred is None or len(pred) == 0:
        return 0.0

    num_preds = len(pred)
    num_gts = len(gt)
    
    cost_matrix = np.zeros((num_preds, num_gts))
    
    for i, p in enumerate(pred):
        p_bbox = p["bbox"]
        p_center = np.array([(p_bbox[0] + p_bbox[2]) / 2, (p_bbox[1] + p_bbox[3]) / 2])
        
        for j, g in enumerate(gt):
            g_bbox = g["bbox"]
            g_center = np.array([(g_bbox[0] + g_bbox[2]) / 2, (g_bbox[1] + g_bbox[3]) / 2])
            
            iou = compute_iou(p_bbox, g_bbox)
            
            dist = np.linalg.norm(p_center - g_center)
            distance_hint = np.exp(-dist / 200) 
            
            label_match = 1.0 if p["label"] == g["label"] else 0.3
            
            # Combine: prioritize IoU, but use distance_hint as a baseline
            combined_geom = max(iou, 0.1 * distance_hint)
            
            score = (0.8 * combined_geom) + (0.2 * label_match)
            cost_matrix[i, j] = 1.0 - score 

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_score_sum = sum(1.0 - cost_matrix[row_ind[k], col_ind[k]] for k in range(len(row_ind)))
    
    # Penalizes both hallucinations and missed objects
    geometry_score = matched_score_sum / max(num_preds, num_gts)
    
    reward = float(np.clip(geometry_score, 0.0, 1.0))
        
    return reward

class BBoxEnv(vf.SingleTurnEnv):

    def __init__(self, dataset, *args, max_size=640, **kwargs):
        self.max_size = max_size
        super().__init__(dataset=dataset, *args, **kwargs)

    def format_dataset(self, dataset: Dataset,system_prompt: str | None = None, category="window"):
        def preprocess_fn(example):
            img = example["image"]
            if not isinstance(img, Image.Image):
                img = Image.open(img)
            img = img.convert("RGB")
            img = smart_resize(img, self.max_size)

            gt = example["ground_truth"]  # bounding box list

            prompt = [{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f'locate every instance that belongs to the following categories: {category}. '
                                f'For each window, report bbox coordinates, in JSON format like this: '
                                f'{{"bbox_2d": [x1, y1, x2, y2], "label": "window"}}'
                    },
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


def load_environment(split="train", **kwargs):
    parser = vf.Parser()
    dataset = load_dataset('UlrickBL/elevation-dataset-synthetic-v2',split=split).shuffle(seed=42)
    rubric = vf.Rubric(
        funcs=[
            reward_parseable,
            reward_geometry_smooth_hungarian
        ],
        weights=[0.2, 0.8]
    )

    return BBoxEnv(
        dataset=dataset,
        parser=parser,
        rubric=rubric,
        **kwargs
    )
