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

ssh-keygen -t ed25519 -C "mail"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

cat ~/.ssh/id_ed25519.pub

Si pb : ssh-keyscan github.com >> ~/.ssh/known_hosts

uv pip install trl==0.19.0
trl vllm-serve --model Qwen/Qwen3-0.6B --max-model-len 5000 --gpu_memory_utilization 0.5

# Install env from code

uv run vf-install semantic -p environments
uv run vf-install rebus_vl_thinking -p environments
uv run vf-install rebus-vl-thinking -p environments
uv run --active vf-install rebus-vl-thinking -p environments
uv run --active vf-install ocr-vl -p environments
uv run --active vf-install reranker-vl -p environments
uv run --active vf-install object-detection-vl -p environments

# Command to run training once everything is installed

source .venv/bin/activate
export PATH="/root/.local/bin:$PATH"

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

NCCL_DEBUG=INFO NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 vf-vllm --model Qwen/Qwen2.5-0.5B-Instruct --max-model-len 15000 --gpu_memory_utilization 0.9

# Qwen 3 VLLM

NCCL_DEBUG=INFO NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen3-0.6B --max-model-len 5000 --gpu_memory_utilization 0.5

# Qwen 2.5 VL VLLM

NCCL_DEBUG=INFO NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen2.5-VL-3B-Instruct --max-model-len 5000 --gpu_memory_utilization 0.5

# Qwen 2.5 VL VLLM with verifiers for proper route

NCCL_DEBUG=INFO NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 vf-vllm --model Qwen/Qwen2.5-VL-7B-Instruct --max-model-len 15000 --gpu_memory_utilization 0.7

NCCL_DEBUG=INFO NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 vf-vllm --model Qwen/Qwen2.5-VL-3B-Instruct --max-model-len 15000 --gpu_memory_utilization 0.9

NCCL_DEBUG=INFO NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 vf-vllm --model Qwen/Qwen3-VL-2B-Instruct --max-model-len 15000 --gpu_memory_utilization 0.9 --dtype half

# GRPO training semantic

CUDA_VISIBLE_DEVICES=1 python src/train_semantic.py
CUDA_VISIBLE_DEVICES=1 python src/train_reranker.py

# GRPO training semantic

CUDA_VISIBLE_DEVICES=1 python src/train_rebus.py 2>&1 | tee logs.txt

CUDA_VISIBLE_DEVICES=1 python src/train_ocr.py 2>&1 | tee logs.txt

CUDA_VISIBLE_DEVICES=1 python src/train_bounding_box.py 2>&1 | tee logs.txt
