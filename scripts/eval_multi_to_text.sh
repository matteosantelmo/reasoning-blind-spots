#!/bin/bash
# Fetch the .env variables
set -a
source .env
set +a

#######################################################
#			  		Frontier Models
#######################################################

# gemini-3.1-flash-lite-preview
python main.py --config-name multi_to_text \
	++solver.model_name=gemini-3.1-flash-lite-preview \
	++solver.backend=google \
	+solver.generate_config.reasoning_effort=medium

# gemini-3-flash-preview
python main.py --config-name multi_to_text \
	++solver.model_name=gemini-3-flash-preview \
	++solver.backend=google \
	+solver.generate_config.reasoning_effort=medium

# gemini-3.1-pro-preview
python main.py --config-name multi_to_text \
	++solver.model_name=gemini-3.1-pro-preview \
	++solver.backend=google \
	+solver.generate_config.reasoning_effort=medium

# gemini-2.5-flash (dynamic reasoning)
python main.py --config-name multi_to_text \
	++solver.model_name=gemini-2.5-flash \
	++solver.backend=google \
	+solver.generate_config.reasoning_tokens=-1

# gemini-2.5-pro (dynamic reasoning)
python main.py --config-name multi_to_text \
	++solver.model_name=gemini-2.5-pro \
	++solver.backend=google \
	+solver.generate_config.reasoning_tokens=-1


# gpt-5
python main.py --config-name multi_to_text \
	++solver.model_name=gpt-5 \
	++solver.backend=openai \
	+solver.generate_config.reasoning_effort=medium \
	++solver.api_key=${OPENAI_API_KEY}

# gpt-5-mini
python main.py --config-name multi_to_text \
	++solver.model_name=gpt-5-mini \
	++solver.backend=openai \
	+solver.generate_config.reasoning_effort=medium \
	++solver.api_key=${OPENAI_API_KEY}

# gpt-5.2
python main.py --config-name multi_to_text \
	++solver.model_name=gpt-5.2 \
	++solver.backend=openai \
	+solver.generate_config.reasoning_effort=medium \
	++solver.api_key=${OPENAI_API_KEY}

# gpt-5.4
python main.py --config-name multi_to_text \
	++solver.model_name=gpt-5.4 \
	++solver.backend=openai \
	+solver.generate_config.reasoning_effort=medium \
	++solver.api_key=${OPENAI_API_KEY}

# gpt-5.4-mini
python main.py --config-name multi_to_text \
	++solver.model_name=gpt-5.4-mini \
	++solver.backend=openai \
	+solver.generate_config.reasoning_effort=medium \
	++solver.api_key=${OPENAI_API_KEY}

# gpt-5.4-nano
python main.py --config-name multi_to_text \
	++solver.model_name=gpt-5.4-nano \
	++solver.backend=openai \
	+solver.generate_config.reasoning_effort=medium \
	++solver.api_key=${OPENAI_API_KEY}

# #######################################################
# #			  Open Source Models on RCP
# #######################################################
# Qwen/Qwen3.5-35B-A3B
python main.py --config-name multi_to_text \
	++solver.model_name=Qwen/Qwen3.5-35B-A3B \
	++solver.backend=openai \
	+solver.base_url=https://inference.rcp.epfl.ch/v1 \
	++solver.api_key=${RCP_OPENAI_API_KEY} \
	+solver.generate_config.max_tokens=32768

# Qwen/Qwen3.5-122B-A10B
python main.py --config-name multi_to_text \
	++solver.model_name=Qwen/Qwen3.5-122B-A10B \
	++solver.backend=openai \
	+solver.base_url=https://inference.rcp.epfl.ch/v1 \
	++solver.api_key=${RCP_OPENAI_API_KEY} \
	+solver.generate_config.max_tokens=32768

# Qwen/Qwen3.5-397B-A17B
python main.py --config-name multi_to_text \
	++solver.model_name=Qwen/Qwen3.5-397B-A17B \
	++solver.backend=openai \
	+solver.base_url=https://inference.rcp.epfl.ch/v1 \
	++solver.api_key=${RCP_OPENAI_API_KEY} \
	+solver.generate_config.max_tokens=32768

# Qwen/Qwen3-VL-30B-A3B-Thinking
python main.py --config-name multi_to_text \
	++solver.model_name=Qwen/Qwen3-VL-30B-A3B-Thinking \
	++solver.backend=openai \
	+solver.base_url=https://inference.rcp.epfl.ch/v1 \
	++solver.api_key=${RCP_OPENAI_API_KEY} \
	+solver.generate_config.max_tokens=32768

# Qwen/Qwen3-VL-235B-A22B-Thinking
python main.py --config-name multi_to_text \
	++solver.model_name=Qwen/Qwen3-VL-235B-A22B-Thinking \
	++solver.backend=openai \
	+solver.base_url=https://inference.rcp.epfl.ch/v1 \
	++solver.api_key=${RCP_OPENAI_API_KEY} \
	+solver.generate_config.max_tokens=32768

# moonshotai/Kimi-K2.6
python main.py --config-name multi_to_text \
	++solver.model_name=moonshotai/Kimi-K2.6 \
	++solver.backend=openai \
	+solver.base_url=https://inference.rcp.epfl.ch/v1 \
	++solver.api_key=${RCP_OPENAI_API_KEY} \
	+solver.generate_config.max_tokens=32768

# moonshotai/Kimi-K2.5
python main.py --config-name multi_to_text \
	++solver.model_name=moonshotai/Kimi-K2.5 \
	++solver.backend=openai \
	+solver.base_url=https://inference.rcp.epfl.ch/v1 \
	++solver.api_key=${RCP_OPENAI_API_KEY} \
	+solver.generate_config.max_tokens=32768

# google/gemma-4-E2B-it
python main.py --config-name multi_to_text \
	++solver.model_name=google/gemma-4-E2B-it \
	++solver.backend=openai \
	+solver.base_url=https://inference.rcp.epfl.ch/v1 \
	++solver.api_key=${RCP_OPENAI_API_KEY} \
	+solver.generate_config.max_tokens=32768 \
	+solver.generate_config.extra_body.chat_template_kwargs.enable_thinking=true

# google/gemma-4-E4B-it
python main.py --config-name multi_to_text \
	++solver.model_name=google/gemma-4-E4B-it \
	++solver.backend=openai \
	+solver.base_url=https://inference.rcp.epfl.ch/v1 \
	++solver.api_key=${RCP_OPENAI_API_KEY} \
	+solver.generate_config.max_tokens=32768 \
	+solver.generate_config.extra_body.chat_template_kwargs.enable_thinking=true

# google/gemma-4-26B-A4B-it
python main.py --config-name multi_to_text \
	++solver.model_name=google/gemma-4-26B-A4B-it \
	++solver.backend=openai \
	+solver.base_url=https://inference.rcp.epfl.ch/v1 \
	++solver.api_key=${RCP_OPENAI_API_KEY} \
	+solver.generate_config.max_tokens=32768 \
	+solver.generate_config.extra_body.chat_template_kwargs.enable_thinking=true

# google/gemma-4-31B-it
python main.py --config-name multi_to_text \
	++solver.model_name=google/gemma-4-31B-it \
	++solver.backend=openai \
	+solver.base_url=https://inference.rcp.epfl.ch/v1 \
	++solver.api_key=${RCP_OPENAI_API_KEY} \
	+solver.generate_config.max_tokens=32768 \
	+solver.generate_config.extra_body.chat_template_kwargs.enable_thinking=true
