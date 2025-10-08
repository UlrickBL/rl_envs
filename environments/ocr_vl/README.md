# ocr-vl

### Overview
- **Environment ID**: `ocr-vl`
- **Short description**: Multilingual numeric OCR VL environment
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
uv run vf-eval ocr-vl   -m gpt-4.1-mini   -n 20 -r 3 -t 1024 -T 0.7   -a '{"lang": "hi","size":1000}'
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments
Document any supported environment arguments and their meaning. Example:

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `lang` | str | `None` | Lang split (if None : all languages) |
| `size` | int | `1000` | Limit on dataset size |

### Metrics
Summarize key metrics your rubric emits and how they’re interpreted.

| Metric | Meaning |
| ------ | ------- |
| `format_reward` | Main scalar reward (weighted sum of criteria) |
| `reward_answer` | 1-CER is mainly digit or non roman sentences, 1-WER of mainly Roman / non digit |

