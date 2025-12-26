import verifiers as vf
from transformers import Qwen3_VLForConditionalGeneration, AutoProcessor

print("running verifier")
model_name = "Qwen/Qwen3-VL-2B-Instruct"
model = Qwen3_VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_name)

model.gradient_checkpointing_enable()
print("model loaded")

vf_env = vf.load_environment(env_id="ulrick-bl/object-detection-vl")

print("env loaded")

args = vf.grpo_defaults(run_name="object-detection-vl")
args.per_device_train_batch_size = 8
args.num_generations = 16
args.gradient_accumulation_steps = 2
args.max_steps = 1000
args.eval_strategy = "steps"
args.eval_steps = 2
args.max_tokens = 1024
args.vllm_server_port= 8000
args.fp16 = True
args.temperature = 0.4
args.learning_rate = 1e-5
args.lr_scheduler_type = "cosine"
args.warmup_steps = 10  

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=processor,
    env=vf_env,
    peft_config=vf.lora_defaults(r=16),
    args=args,
)
trainer.train()