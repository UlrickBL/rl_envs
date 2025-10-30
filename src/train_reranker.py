import verifiers as vf
from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import wandb

print("running verifier")
model_name = "Qwen/Qwen3-VL-2B-Instruct"
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_name)

model.gradient_checkpointing_enable()
print("model loaded")

vf_env = vf.load_environment(env_id="reranker-vl",max_docs="4",max_size=650)

print("env loaded")

args = vf.grpo_defaults(run_name="reranker-vl")
args.per_device_train_batch_size = 8
args.num_generations = 8
args.gradient_accumulation_steps = 8
args.max_steps = 1000
args.eval_strategy = "steps"
args.eval_steps = 50
args.max_tokens = 1024
args.vllm_server_port= 8000
args.fp16 = True
args.temperature = 0.7
args.learning_rate = 3e-5
args.lr_scheduler_type = "cosine"
args.warmup_steps = 30 
args.beta = 0.01
args.save_strategy = "steps"
args.save_steps = 50
args.output_dir = "outputs/reranker-vl"

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=processor,
    env=vf_env,
    peft_config=vf.lora_defaults(r=16),
    args=args,
)
trainer.train()