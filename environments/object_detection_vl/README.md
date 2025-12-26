# object-detection-vl

### Overview
- **Environment ID**: `object-detection-vl`
- **Short description**: Object detection / bounding box env for VLM models (Qwen VL like)
- **Tags**: Vision, Bounding box, Object detection, Bbox

### Datasets
- **Primary dataset(s)**: Currently working on it. Custom dataset of images with annotated 2D bounding boxes  
  Format example:
  ```json
  [
    {"bbox_2d": [607, 754, 639, 810], "label": "window"},
    {"bbox_2d": [123, 229, 155, 285], "label": "window"}
  ]
- **Split sizes**:

### Task
- **Type**: single-turn
- **Parser**: XMLParser
- **Rubric overview**:  Evaluates prediction quality based on:
- Parse correctness
- Object count accuracy
- Order-invariant matching (Hungarian algorithm)
- Bounding box IoU
- Label correctness

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval object-detection-vl
```

Configure model and sampling:

```bash
uv run vf-eval ocr-vl   -m gpt-4.1-mini   -n 20 -r 3 -t 1024 -T 0.7 
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `size` | int | `None` | Limit dataset size |
| `max_size` | int | `640` | Longest dimension for smart image resize |
| `iou_weight` | float | `0.7` | Weight assigned to IoU in matching score |
| `label_weight` | float | `0.3` | Weight assigned to label correctness |


### Metrics
| Metric | Meaning |
| ------ | ------- |
| `format_reward` | Whether output successfully parsed into a detection list |
| `count_reward` | Closeness of predicted vs GT number of objects |
| `bbox_reward` | IoU-based Hungarian-matched bounding box localization reward |
| `label_reward` | Hungarian-matched label correctness |
| `final_reward` | Main scalar reward (weighted sum of all components) |


