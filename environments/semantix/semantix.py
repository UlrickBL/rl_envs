import verifiers as vf
import random
from wordfreq import top_n_list
from sentence_transformers import SentenceTransformer, util
from verifiers.types import Messages, State
from typing import List, Tuple

NOTHINK_GUESS_SYSTEM_PROMPT = """You are a competitive game player. \
Make sure you read the game instructions carefully, and always follow the required format.

In each turn, give only your guess inside <guess>...</guess> tags."""

def get_random_word(lang="en", n=5000):
    words = top_n_list(lang, n)
    return random.choice(words)

def count_turns_reward_func(parser, completion, answer, **kwargs) -> float:
    num_turns = len([x for x in completion if x["role"] == "assistant"])
    is_correct = check_answer_reward_func(parser, completion, answer, **kwargs)
    return is_correct / (num_turns + 1)

def get_similarity(model,ground_truth,guess) :
    embeddings = model.encode([ground_truth, guess], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    return similarity

class SemantixEnv(vf.MultiTurnEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def is_completed(self, messages: Messages, state: State, **kwargs) -> bool:
        assert isinstance(messages, list)
        assistant_count = len([m for m in messages if m["role"] == "assistant"])
        num_turns = state["info"]["num_turns"]
        return assistant_count >= num_turns

    def env_response(self, messages: Messages, state: State, **kwargs) -> Tuple[Messages, State]:
        assert isinstance(messages, list)
        assistant_count = len([m for m in messages if m["role"] == "assistant"])
        num_turns = state["info"]["num_turns"]

        if assistant_count < num_turns:
            follow_ups = state["info"]["follow_ups"]
            follow_up_idx = assistant_count - 1

            if follow_up_idx < len(follow_ups):
                return [{"role": "user", "content": follow_ups[follow_up_idx]}], state

        return [{"role": "user", "content": "Continue"}], state
        
def load_environment(model_name = "paraphrase-multilingual-MiniLM-L12-v2",
                     lang="en") -> vf.Environment:
    '''
    Loads environment for semantic
    '''
    model = SentenceTransformer(model_name)
    ground_truth = get_random_word(lang=lang)

    system_prompt = NOTHINK_GUESS_SYSTEM_PROMPT
    
    parser = vf.XMLParser(fields=["guess"], answer_field="guess")
    rubric = vf.Rubric(parser=parser)
    rubric.add_reward_func(check_answer_reward_func)
    rubric.add_reward_func(parser.get_format_reward_func(), weight=0.2)
    
    raise NotImplementedError("Implement your custom environment here.")
