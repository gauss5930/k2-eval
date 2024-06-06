#!/bin/bash

# Usage: ./script.sh
# This script processes a predefined list of models.

# List of models
declare -a MODELS=(
    "gpt-4o"
)


# Iterate over each model in the list
for MODEL_PATH in "${MODELS[@]}"
do
    echo "Running text generation for model: $MODEL_PATH"
    python text_generation.py --model_path eval_model --model_type openai

    echo "Evaluating model: $MODEL_PATH"
    python evaluate.py eval_model
done