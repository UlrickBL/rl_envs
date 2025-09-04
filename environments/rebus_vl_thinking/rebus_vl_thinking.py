import verifiers as vf

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

def load_environment(**kwargs) -> vf.Environment:
    '''
    Loads a custom environment.
    '''
    raise NotImplementedError("Implement your custom environment here.")
