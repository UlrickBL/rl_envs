import verifiers as vf
from datasets import load_dataset
from datasets import Dataset
from verifiers.types import ChatMessage
from io import BytesIO
import base64
from PIL import Image

SYSTEM_PROMPT = """You are a competitive game player playing with a game master. \
Make sure you read the game instructions carefully, and always follow the required format.

In each turn, give only your guess inside <guess>...</guess> tags.
"""

GAME_PROMPT = """You are playing a rebus game.  
In this game, you are given a picture representing a rebus puzzle.  
The rebus is a visual encoding of a French word or short expression.  

Your task:  
1. Carefully analyze the image.  
2. Think step by step about the possible interpretation of the visual clues.  
3. Deduce the French word or expression that matches the rebus.  

Important:  
- You should write out your reasoning process so we can see how you think.  
- At the very end, output only your final solution inside XML-like tags:  

<guess>your_final_answer_here</guess>  

Do not include explanations or extra text after the <guess> block.  
"""

class RebusEnv(vf.SingleTurnEnv):
    """
    Environment for Rebus Guessing
    """
    def format_dataset(
            self,
            dataset: Dataset,
            system_prompt: str | None = None,
            answer_key: str = "word",
            image_key: str = "image"
        ) -> Dataset:

        def image_to_base64(image: Image, format='png'):
            buffered = BytesIO()
            image.save(buffered, format=format)
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        def format_prompt_fn(image : Image) -> list[ChatMessage]:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]})
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": GAME_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_to_base64(image)}"}}
                ]
            })
            return messages
        
        def preprocess_fn(example):
            return {
                "prompt": format_prompt_fn(example["image"]),
                "answer": example["word"],
            }
        return dataset.map(preprocess_fn)

def reward_answer(parser, completion, answer):
    response = parser.parse_answer(completion) or ''
    return response == answer

def load_environment(num_examples=None, **kwargs) -> vf.Environment:
    '''
    Loads Rebus Environment.
    '''
    dataset = load_dataset("UlrickBL/rebus_french_rl")
    parser = vf.XMLParser(fields=["guess"], answer_field="guess")
    rubric = vf.Rubric(
        funcs=[reward_answer, parser.get_format_reward_func()],
        weights=[1.0, 0.2]
    )

    return RebusEnv(
        dataset=dataset,
        system_prompt=SYSTEM_PROMPT,
        parser=parser,
        rubric=rubric,
        **kwargs
    )
