#!/bin/bash
# Fetch the .env variables
set -a
source .env
set +a

###################################
#	Frontier Models with Tools
###################################

# gemini-3.1-flash-lite
python main.py --config-name text_only_tools \
	++solver.model_name=gemini-3.1-flash-lite-preview \
	++solver.backend=google \
	+solver.generate_config.reasoning_effort=medium \
	++solver.tools.provides=google

# gemini-3.1-pro
python main.py --config-name text_only_tools \
	++solver.model_name=gemini-3.1-pro-preview \
	++solver.backend=google \
	+solver.generate_config.reasoning_effort=medium \
	++solver.tools.provides=google

# gpt-5.4-mini
python main.py --config-name text_only_tools \
	++solver.model_name=gpt-5.4-mini \
	++solver.backend=openai \
	+solver.generate_config.reasoning_effort=medium \
	++solver.tools.provides=openai

# gpt-5.4
python main.py --config-name text_only_tools \
	++solver.model_name=gpt-5.4 \
	++solver.backend=openai \
	+solver.generate_config.reasoning_effort=medium \
	++solver.tools.provides=openai


###################################
#	OSS Models with Tools
###################################

# Qwen3.5-397B-A17B
python main.py --config-name text_only_tools \
	++solver.model_name=Qwen/Qwen3.5-397B-A17B \
	++solver.backend=openai \
	+solver.base_url=https://inference.rcp.epfl.ch/v1 \
	++solver.api_key=${OSS_INFERENCE_OPENAI_API} \
	+solver.generate_config.max_tokens=32768 \
	sandbox=docker

# Qwen3-VL-30B-A3B-Thinking
python main.py --config-name text_only_tools \
	++solver.model_name=Qwen/Qwen3-VL-30B-A3B-Thinking \
	++solver.backend=openai \
	+solver.base_url=https://inference.rcp.epfl.ch/v1 \
	++solver.api_key=${OSS_INFERENCE_OPENAI_API} \
	+solver.generate_config.max_tokens=32768 \
	sandbox=docker

# gpt-oss-120b
python main.py --config-name text_only_tools \
	++solver.model_name=openai/gpt-oss-120b \
	++solver.backend=openai \
	+solver.base_url=https://inference.rcp.epfl.ch/v1 \
	++solver.api_key=${OSS_INFERENCE_OPENAI_API} \
	+solver.generate_config.reasoning_effort=medium \
	sandbox=docker

# Kimi-K2.5
python main.py --config-name text_only_tools \
	++solver.model_name=moonshotai/Kimi-K2.5 \
	++solver.backend=openai \
	+solver.base_url=https://inference.rcp.epfl.ch/v1 \
	++solver.api_key=${OSS_INFERENCE_OPENAI_API} \
	+solver.generate_config.max_tokens=32768 \
	sandbox=docker

# Kimi-K2.6
python main.py --config-name text_only_tools \
	++solver.model_name=moonshotai/Kimi-K2.6 \
	++solver.backend=openai \
	+solver.base_url=https://inference.rcp.epfl.ch/v1 \
	++solver.api_key=${OSS_INFERENCE_OPENAI_API} \
	+solver.generate_config.max_tokens=32768 \
	sandbox=docker

# GLM-5.1
python main.py --config-name text_only_tools \
	++solver.model_name=zai-org/GLM-5.1 \
	++solver.backend=openai \
	+solver.base_url=https://inference.rcp.epfl.ch/v1 \
	++solver.api_key=${OSS_INFERENCE_OPENAI_API} \
	+solver.generate_config.max_tokens=32768 \
	sandbox=docker

# GLM-5.2
python main.py --config-name text_only_tools \
	++solver.model_name=zai-org/GLM-5.2 \
	++solver.backend=openai \
	+solver.base_url=https://inference.rcp.epfl.ch/v1 \
	++solver.api_key=${OSS_INFERENCE_OPENAI_API} \
	+solver.generate_config.max_tokens=32768 \
	sandbox=docker
