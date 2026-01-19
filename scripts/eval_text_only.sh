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


#######################################################
#			  Open Source Models on RCP
#######################################################
# Qwen3-VL-235B-A22B-Thinking
python main.py --config-name text_only \
	++solver.model_name=Qwen/Qwen3-VL-235B-A22B-Thinking \
	++solver.backend=openai \
	+solver.base_url=https://inference.rcp.epfl.ch/v1 \
	++solver.api_key=${RCP_OPENAI_API_KEY}

# gpt-oss-120b
python main.py --config-name text_only \
	++solver.model_name=openai/gpt-oss-120b \
	++solver.backend=openai \
	+solver.base_url=https://inference.rcp.epfl.ch/v1 \
	++solver.api_key=${RCP_OPENAI_API_KEY}
