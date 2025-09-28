# ocr-vl

> Replace the placeholders below, then remove this callout.

### Overview
- **Environment ID**: `ocr-vl`
- **Short description**: OCR VL environment
- **Tags**: Vision, OCR, VLM

### Datasets
- **Primary dataset(s)**: webui_multilingual_ocr multilingual WEB UI images and transcriptions
- **Source links**: [<links>](https://huggingface.co/datasets/deepcopy/webui_multilingual_ocr)
- **Split sizes**: train 307k

### Task
- **Type**: single-turn
- **Parser**: XMLParser
- **Rubric overview**: mix of WER and CER

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval ocr-vl
```

Configure model and sampling:

```bash
uv run vf-eval ocr-vl   -m gpt-4.1-mini   -n 20 -r 3 -t 1024 -T 0.7   -a '{"key": "value"}'  # env-specific args as JSON
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments
Document any supported environment arguments and their meaning. Example:

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `foo` | str | `"bar"` | What this controls |
| `max_examples` | int | `-1` | Limit on dataset size (use -1 for all) |

### Metrics
Summarize key metrics your rubric emits and how they’re interpreted.

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar reward (weighted sum of criteria) |
| `accuracy` | Exact match on target answer |

