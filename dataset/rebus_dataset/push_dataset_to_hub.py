from huggingface_hub import login
from datasets import load_from_disk

login()

dataset = load_from_disk("rebus_french_rl")

dataset.push_to_hub("UlrickBL/rebus_french_rl")
