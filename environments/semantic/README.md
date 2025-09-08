# semantic

### Overview
- **Environment ID**: `semantic`
- **Short description**: Semantic is a multi-turn reinforcement learning environment inspired by the popular Cemantix game. At the start of each episode, the environment samples a target word in a given language. The agent must iteratively guess the word, receiving feedback in the form of semantic similarity scores computed via an embedding model. This setup encourages discovery through exploration, supports multilingual evaluation, and promotes synonym and semantic grounding. Unlike classic next-token prediction, the environment provides a masked semantic objective, making it suitable for distillation-style training of large language models (akin to a BERT-like signal) within an RL framework. The RL env offer a huge flexibility and poor reward hacking (number of turns, similarity models, input words, ...).
- **Tags**: semantic, cemantix, multiturn, BERT, xml

### Datasets
- **Primary dataset(s)**: wordfreq
- **Source links**: https://github.com/rspeer/wordfreq/
- **Split sizes**: No split because sampled at runtime

### Task
- **Type**: multi-turn (game interaction)
- **Parser**: XMLParser with guess
- **Rubric overview**: At each turn, the agent receives a similarity-based reward (0 for format errors, ∈[0,1] otherwise, and 2 for an exact match), scaled by an exponential decay with the turn number and normalized by the maximum achievable reward.

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval semantix
```

Configure model and sampling:

```bash
uv run vf-eval semantix   -m gpt-4.1-mini   -n 20 -r 3 -t 1024 -T 0.7   -a '{"model_name": "paraphrase-multilingual-MiniLM-L12-v2","lang":"en","dataset_len":10000}'  

```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments
Document any supported environment arguments and their meaning. Example:

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `model_name` | str | `"paraphrase-multilingual-MiniLM-L12-v2"` | Semantic similarity model |
| `lang` | str | `"en"` | Language for words sampling and guessing |
| `dataset_len` | int | `10000` | Number of sampled words in the dataset |

### Metrics
Summarize key metrics your rubric emits and how they’re interpreted.

| Metric | Meaning |
| ------ | ------- |
| `reward` | At each turn, the agent receives a similarity-based reward (0 for format errors, ∈[0,1] otherwise, and 2 for an exact match), scaled by an exponential decay with the turn number and normalized by the maximum achievable reward |

