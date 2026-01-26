#!/bin/bash
# Fetch the .env variables
set -a
source .env
set +a

#######################################################
#			  		Frontier Models
#######################################################
# gemini-2.5-flash (dynamic reasoning)
python main.py --config-name text_only \
	++solver.model_name=gemini-2.5-flash \
	++solver.backend=google \
	+solver.generate_config.reasoning_tokens=-1

# gemini-2.5-pro (dynamic reasoning)
python main.py --config-name text_only \
	++solver.model_name=gemini-2.5-pro \
	++solver.backend=google \
	+solver.generate_config.reasoning_tokens=-1

# gemini-3-flash-preview
python main.py --config-name text_only \
	++solver.model_name=gemini-3-flash-preview \
	++solver.backend=google \
	+solver.generate_config.reasoning_effort=medium

# gemini-3-pro-preview
python main.py --config-name text_only \
	++solver.model_name=gemini-3-pro-preview \
	++solver.backend=google \
	+solver.generate_config.reasoning_effort=medium

# gpt-5
python main.py --config-name text_only \
	++solver.model_name=gpt-5 \
	++solver.backend=openai \
	+solver.generate_config.reasoning_effort=medium \
	++solver.api_key=${OPENAI_API_KEY}

# o3
python main.py --config-name text_only \
	++solver.model_name=o3 \
	++solver.backend=openai \
	+solver.generate_config.reasoning_effort=medium \
	++solver.api_key=${OPENAI_API_KEY}

# gpt-5.2
python main.py --config-name text_only \
	++solver.model_name=gpt-5.2 \
	++solver.backend=openai \
	+solver.generate_config.reasoning_effort=medium \
	++solver.api_key=${OPENAI_API_KEY}

# gpt-5-mini
python main.py --config-name text_only \
	++solver.model_name=gpt-5-mini \
	++solver.backend=openai \
	+solver.generate_config.reasoning_effort=medium \
	++solver.api_key=${OPENAI_API_KEY}

# gpt-4.1
python main.py --config-name text_only \
	++solver.model_name=gpt-4.1 \
	++solver.backend=openai \
	+solver.generate_config.reasoning_effort=medium \
	++solver.api_key=${OPENAI_API_KEY}

#######################################################
#			  Open Source Models on RCP
#######################################################
# gpt-oss-120b
python main.py --config-name text_only \
	++solver.model_name=openai/gpt-oss-120b \
	++solver.backend=openai \
	+solver.base_url=https://inference.rcp.epfl.ch/v1 \
	++solver.api_key=${RCP_OPENAI_API_KEY} \
	+solver.generate_config.reasoning_effort=medium \
	+solver.generate_config.max_tokens=32768

# gpt-oss-20b
python main.py --config-name text_only \
	++solver.model_name=openai/gpt-oss-20b \
	++solver.backend=openai \
	+solver.base_url=https://inference.rcp.epfl.ch/v1 \
	++solver.api_key=${RCP_OPENAI_API_KEY} \
	+solver.generate_config.reasoning_effort=medium \
	+solver.generate_config.max_tokens=32768

# Qwen3-VL-30B-A3B-Instruct
python main.py --config-name text_only \
	++solver.model_name=Qwen/Qwen3-VL-30B-A3B-Instruct \
	++solver.backend=openai \
	+solver.base_url=https://inference.rcp.epfl.ch/v1 \
	++solver.api_key=${RCP_OPENAI_API_KEY} \
	+solver.generate_config.max_tokens=32768

# Qwen3-VL-30B-A3B-Thinking
python main.py --config-name text_only \
	++solver.model_name=Qwen/Qwen3-VL-30B-A3B-Thinking \
	++solver.backend=openai \
	+solver.base_url=https://inference.rcp.epfl.ch/v1 \
	++solver.api_key=${RCP_OPENAI_API_KEY} \
	+solver.generate_config.max_tokens=32768

# Qwen3-VL-235B-A22B-Instruct
python main.py --config-name text_only \
	++solver.model_name=Qwen/Qwen3-VL-235B-A22B-Instruct \
	++solver.backend=openai \
	+solver.base_url=https://inference.rcp.epfl.ch/v1 \
	++solver.api_key=${RCP_OPENAI_API_KEY} \
	+solver.generate_config.max_tokens=32768

# Qwen3-VL-235B-A22B-Thinking
python main.py --config-name text_only \
	++solver.model_name=Qwen/Qwen3-VL-235B-A22B-Thinking \
	++solver.backend=openai \
	+solver.base_url=https://inference.rcp.epfl.ch/v1 \
	++solver.api_key=${RCP_OPENAI_API_KEY} \
	+solver.generate_config.max_tokens=32768

# Apertus-70B-Instruct-2509
python main.py --config-name text_only \
	++solver.model_name=swiss-ai/Apertus-70B-Instruct-2509 \
	++solver.backend=openai \
	+solver.base_url=https://inference.rcp.epfl.ch/v1 \
	++solver.api_key=${RCP_OPENAI_API_KEY} \
	+solver.generate_config.max_tokens=32768

# Llama-3.3-70B-Instruct
python main.py --config-name text_only \
	++solver.model_name=meta-llama/Llama-3.3-70B-Instruct \
	++solver.backend=openai \
	+solver.base_url=https://inference.rcp.epfl.ch/v1 \
	++solver.api_key=${RCP_OPENAI_API_KEY} \
	+solver.generate_config.max_tokens=32768

# Llama-4-Maverick-17B-128E-Instruct
python main.py --config-name text_only \
	++solver.model_name=meta-llama/Llama-4-Maverick-17B-128E-Instruct \
	++solver.backend=openai \
	+solver.base_url=https://inference.rcp.epfl.ch/v1 \
	++solver.api_key=${RCP_OPENAI_API_KEY} \
	+solver.generate_config.max_tokens=32768

# DeepSeek-V3.2
python main.py --config-name text_only \
	++solver.model_name=deepseek-ai/DeepSeek-V3.2 \
	++solver.backend=openai \
	+solver.base_url=https://inference.rcp.epfl.ch/v1 \
	++solver.api_key=${RCP_OPENAI_API_KEY} \
	+solver.generate_config.max_tokens=32768

# DeepSeek-V3.2-Speciale
python main.py --config-name text_only \
	++solver.model_name=deepseek-ai/DeepSeek-V3.2-Speciale \
	++solver.backend=openai \
	+solver.base_url=https://inference.rcp.epfl.ch/v1 \
	++solver.api_key=${RCP_OPENAI_API_KEY} \
	+solver.generate_config.max_tokens=32768

# Mistral-Large-3-675B-Instruct-2512
python main.py --config-name text_only \
	++solver.model_name=mistralai/Mistral-Large-3-675B-Instruct-2512 \
	++solver.backend=openai \
	+solver.base_url=https://inference.rcp.epfl.ch/v1 \
	++solver.api_key=${RCP_OPENAI_API_KEY} \
	+solver.generate_config.max_tokens=32768
