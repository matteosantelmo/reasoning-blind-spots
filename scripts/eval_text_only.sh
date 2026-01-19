#!/bin/bash

# Reproducibility script for evaluating text-only questions with different LLMs.
## Gemini-2.5-flash (dynamic reasoning)
python main.py --config-name text_only \
	++solver.model_name=gemini-2.5-flash \
	++solver.backend=google \
	+solver.generate_config.reasoning_tokens=-1
