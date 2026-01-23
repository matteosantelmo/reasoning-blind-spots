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

# Gemini 2.5 Flash Image (Nano Banana - fast, efficient)
python main.py --config-name image_gen \
	++solver.model_name=gemini-2.5-flash-image \
	++solver.backend=google

# # Gemini 3 Pro Image Preview (Nano Banana Pro - professional quality with thinking)
# python main.py --config-name image_gen \
# 	++solver.model_name=gemini-3-pro-image-preview \
# 	++solver.backend=google

# #######################################################
# #                OpenAI Image Models
# #######################################################

# # GPT Image 1 (high quality)
# python main.py --config-name image_gen \
# 	++solver.model_name=gpt-image-1 \
# 	++solver.backend=openai \
# 	++solver.quality=high

# # GPT Image 1 Mini (faster, lower cost)
# python main.py --config-name image_gen \
# 	++solver.model_name=gpt-image-1-mini \
# 	++solver.backend=openai \
# 	++solver.quality=auto

# # DALL-E 3
# python main.py --config-name image_gen \
# 	++solver.model_name=dall-e-3 \
# 	++solver.backend=openai \
# 	++solver.size=1024x1024

# #######################################################
# #                Google Imagen Models
# #######################################################

# # Imagen 4.0 Generate
# python main.py --config-name image_gen \
# 	++solver.model_name=imagen-4.0-generate-001 \
# 	++solver.backend=google

# # Imagen 4.0 Fast Generate
# python main.py --config-name image_gen \
# 	++solver.model_name=imagen-4.0-fast-generate-001 \
# 	++solver.backend=google
