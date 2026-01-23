#!/bin/bash
# Evaluation script for image generation tasks
# This script runs the benchmark for text-to-image and image-gen question types

# Fetch the .env variables
set -a
source .env
set +a

#######################################################
#          Google Gemini Image Models (Recommended)
#######################################################

# Gemini 2.5 Flash Image (Nano Banana)
python main.py --config-name image_gen \
	++solver.model_name=gemini-2.5-flash-image \
	++solver.backend=google

# Gemini 3 Pro Image Preview (Nano Banana Pro)
python main.py --config-name image_gen \
	++solver.model_name=gemini-3-pro-image-preview \
	++solver.backend=google

# #######################################################
# #                OpenAI Image Models
# #######################################################

# GPT Image 1.5
python main.py --config-name image_gen \
	++solver.model_name=gpt-image-1.5 \
	++solver.backend=openai \
	++solver.quality=medium

# GPT Image 1
python main.py --config-name image_gen \
	++solver.model_name=gpt-image-1 \
	++solver.backend=openai \
	++solver.quality=medium

# GPT Image 1 Mini
python main.py --config-name image_gen \
	++solver.model_name=gpt-image-1-mini \
	++solver.backend=openai \
	++solver.quality=auto
