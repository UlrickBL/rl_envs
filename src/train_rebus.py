import verifiers as vf

print("running verifier")
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
model, tokenizer = vf.get_model_and_tokenizer(model_name)
model.gradient_checkpointing_enable()
print("model loaded")

vf_env = vf.load_environment(env_id="rebus_vl_thinking")
print("env loaded")

args = vf.grpo_defaults(run_name="rebus_vl_thinking")
args.per_device_train_batch_size = 2
args.num_generations = 16
args.gradient_accumulation_steps = 8
args.max_steps = 100
args.eval_strategy = "steps"
args.eval_steps = 2
args.max_tokens = 1024
args.vllm_server_port= 8000
args.fp16 = True

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    peft_config=vf.lora_defaults(r=4),
    args=args,
)
trainer.train()