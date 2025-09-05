import verifiers as vf

print("running verifier")
model_name = "Qwen/Qwen3-0.6B"
model, tokenizer = vf.get_model_and_tokenizer(model_name)
print("model loaded")

vf_env = vf.load_environment(env_id="semantic")
print("env loaded")

args = vf.grpo_defaults(run_name="semantic")
args.per_device_train_batch_size = 12
args.num_generations = 12
args.gradient_accumulation_steps = 8
args.max_steps = 100
args.eval_strategy = "steps"
args.eval_steps = 2
args.max_tokens = 1024
args.vllm_server_port= 4227

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    peft_config=vf.lora_defaults(),
    args=args,
)
trainer.train()