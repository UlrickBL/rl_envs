# Command to install

curl -LsSf https://astral.sh/uv/install.sh | sh
uv init
uv add 'verifiers[all]' && uv pip install flash-attn --no-build-isolation

source .venv/bin/activate

rm -Rf .venv
uv venv .venv
uv pip install -e .

export PATH="/root/.local/bin:$PATH"

# For github (to pull code on GPU cloud)

ssh-keygen -t ed25519 -C "your_email@example.com"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

cat ~/.ssh/id_ed25519.pub

uv pip install trl==0.19.0
trl vllm-serve --model Qwen/Qwen3-0.6B --max-model-len 5000 --gpu_memory_utilization 0.5

# Install env from code

uv run vf-install semantic -p environments
uv run vf-install rebus_vl_thinking -p environments

# Command to run training once everything is installed 

source .venv/bin/activate
export PATH="/root/.local/bin:$PATH"

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1 

export CUDA_VISIBLE_DEVICES=0,1

# Qwen 2.5 VLLM
NCCL_DEBUG=INFO NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 vf-vllm --model Qwen/Qwen2.5-0.5B-Instruct --max-model-len 15000 --gpu_memory_utilization 0.7

# Qwen 3 VLLM
NCCL_DEBUG=INFO NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen3-0.6B --max-model-len 5000 --gpu_memory_utilization 0.5

# Qwen 2.5 VL VLLM
NCCL_DEBUG=INFO NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen2.5-VL-3B-Instruct --max-model-len 5000 --gpu_memory_utilization 0.5

# GRPO training semantic
CUDA_VISIBLE_DEVICES=1 python src/train_semantic.py

# GRPO training semantic
CUDA_VISIBLE_DEVICES=1 python src/train_rebus.py