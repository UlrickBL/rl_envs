# semantix_rl_env

curl -LsSf https://astral.sh/uv/install.sh | sh
uv init
uv add 'verifiers[all]' && uv pip install flash-attn --no-build-isolation


uv run vf-install semantix -p environments