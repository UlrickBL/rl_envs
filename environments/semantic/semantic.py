import verifiers as vf
import random
from wordfreq import top_n_list
from sentence_transformers import SentenceTransformer, util
from verifiers.types import Messages, State
from typing import List, Tuple
import math
from datasets import Dataset, load_dataset

ALPHA = 0.5
MAX_TURN = 15

NOTHINK_GUESS_SYSTEM_PROMPT = """You are a competitive game player playing with a game master. \
Make sure you read the game instructions carefully, and always follow the required format.

In each turn, give only your guess inside <guess>...</guess> tags."""

GAME_PROMPT = """You are playing a word similarity guessing game.

Rules:
- The game is in the language: {LANGUAGE}.
- A secret target word has been chosen by the system.
- Your task is to guess this word.
- After each guess, you will receive from the game master:
  - A similarity score (0 to 1) between your guess and the target word.
  - A list of the top 10 closest words (with their similarity scores) you already gave.
- You must use this feedback to make your next guess, refining step by step, until you discover the target word.
- You can think each attempt but finish with a SINGLE GUESS PER attempt, in the correct language inside <guess>...</guess> tags (and wait for the game master to give you the similarity before).
- Give on word per guess and wait for the game master feedback.

Start by guessing a firt word quickly:
"""

TURN_PROMPT = """Your last guess was: "{GUESS}"
Similarity score with the target word: {SCORE}

Here are the top 10 closest words to your guess:
{TOP_WORDS}

Based on this feedback, provide your next single-word guess in {LANGUAGE}.
"""

def prepare_dataset(dataset_len,lang) :
    data = []
    for i in range(dataset_len) :
        data.append(
            {
                "prompt": [{"role": "system", "content": NOTHINK_GUESS_SYSTEM_PROMPT},{"role": "user", "content": GAME_PROMPT.format(LANGUAGE=lang)}],
                "info": {
                    "ground_truths": get_random_word(lang=lang),
                    "turn_num": 0,
                    "guesses": [],
                    "similarities": [],
                    "language" : lang
                },
            }
        )
    return  Dataset.from_list(data)

def get_random_word(lang="en", n=5000):
    words = top_n_list(lang, n)
    return random.choice(words)

def create_weighted_rewards(): # TODO
    """
    base_reward = 0 if format issue, similarity if not reached (so between 0 and 1) and 2 if proper word found
    reward=base_reward * exp(-α*turn)
    weight it with the max reward available
    """
    def weighted_reward(completion, state, **kwargs):
        actual_turns = state["info"]["turn_num"]
        best_similarity = max(state["info"]["similarities"])
        base_reward = 0
        if state["info"]["guesses"][-1] == state["info"]["ground_truths"] :
            base_reward = 2
        else :
            base_reward = best_similarity
        return base_reward * math.exp(-ALPHA*actual_turns)/2
    return weighted_reward


def get_similarity(model,ground_truth,guess) :
    embeddings = model.encode([ground_truth, guess], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    return similarity

class SemantixEnv(vf.MultiTurnEnv):
    """
    Objective :
    For X turn :
        - The LLM give a word
        - We check if it is the current one (with SequenceMatch)
            - If yes : reward it with the format and the number of turn
            - If no : answer with the similarity score (+ the top 10 words previously sent for long term memory?)
    """
    def __init__(self,similarity_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.similarity_model = similarity_model

    def is_completed(self, messages: Messages, state: State, **kwargs) -> bool:
        """
        Multiturn is completed when 
        - Number of turn is reached
        - Correct word is sent
        """
        assert isinstance(messages, list)
        assistant_count = len([m for m in messages if m["role"] == "assistant"])
        num_turns = state["info"]["turn_num"]
        
        if assistant_count >= MAX_TURN :
            return True
        else :
            assistant_msgs = [m["content"] for m in messages if m["role"] == "assistant"]
            if state["info"]["turn_num"] - 1 > 0 :
                guess = self.parser.parse_answer(assistant_msgs[state["info"]["turn_num"] - 1])
                return guess == state["info"]["ground_truths"]
            else :
                return False

    def env_response(self, messages: Messages, state: State, **kwargs) -> Tuple[Messages, State]:
        """
        Env response is basically TURN_PROMPT completed with the correct values
        """
        assistant_msgs = [m["content"] for m in messages if m["role"] == "assistant"]
        guess = self.parser.parse_answer(assistant_msgs[state["info"]["turn_num"] - 1])
        similarity = get_similarity(self.similarity_model,state["info"]["ground_truths"],guess)
        
        state["info"]["turn_num"] += 1
        state["info"]["guesses"].append(guess)
        state["info"]["similarities"].append(similarity)
        
        pairs = list(zip(state["info"]["guesses"], state["info"]["similarities"]))
        top_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)[:10]
        top_words_str = "\n".join([f"{w}: {s:.2f}" for w, s in top_pairs])
        
        return [{"role": "user", "content": TURN_PROMPT.format(GUESS=guess,SCORE=similarity,TOP_WORDS=top_words_str,LANGUAGE=state["info"]["language"])}], state
        
def load_environment(model_name = "paraphrase-multilingual-MiniLM-L12-v2",
                     lang="en",
                     max_turns=30,
                     dataset_len=10000) -> vf.Environment:
    '''
    Loads environment for semantic
    '''
    similarity_model = SentenceTransformer(model_name)
    
    words_dataset = prepare_dataset(dataset_len,lang)

    system_prompt = NOTHINK_GUESS_SYSTEM_PROMPT
    
    parser = vf.XMLParser(fields=["guess"], answer_field="guess")
    rubric = vf.Rubric(parser=parser,funcs=[create_weighted_rewards()], weights=[1.0])
    
    env_instance = SemantixEnv(dataset=words_dataset, rubric=rubric, parser=parser,max_turns=max_turns, system_prompt=system_prompt, similarity_model=similarity_model)
    
    return env_instance
