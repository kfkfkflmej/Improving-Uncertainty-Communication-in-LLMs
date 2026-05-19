
models=(
'google/gemma-3-4b-it'
'google/gemma-3-12b-it'
)

datasets=(
    'nq_open_200'
    'simple_qa_200'
    'simple_qa_art'
    'simple_qa_st'
    'simple_qa_geography'
)


for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        model_name=$(basename "$model")
        data_path="evaluating_LC/datasets/${dataset}.csv"

        # Execution for base models (uses "vanilla")
        PYTHONPATH=lib python3 bin/linguistic_confidence.py \
            --model "$model" \
            --data_path "$data_path" \
            --dataset "$dataset" \
            --temperature 1.0 \
            --prompt_style "vanilla"

        # Execution for base models (uses "vanilla_uncertainty")
        if [[ "$model" == google/* ]]; then
            PYTHONPATH=lib python3 bin/linguistic_confidence.py\
                --model "$model" \
                --data_path "$data_path" \
                --dataset "$dataset" \
                --temperature 1.0 \
                --prompt_style "vanilla_uncertainty"
        fi

        # Execution for fine-tuned models (uses "vanilla")
        if [[ "$model" =~ ([0-9]+)b ]]; then
            model_size="${BASH_REMATCH[1]}"
        else
            model_size="unknown"
        fi
        if [[ "$model_size" == "12" ]]; then
                lora_path="pashadohnal/gemma-3-12b-uncertain-s-t"
            else
                lora_path="pashadohnal/gemma-3-4b-uncertain-s-t"
        fi
        PYTHONPATH=lib python3 bin/linguistic_confidence.py\
            --model "$model" \
            --data_path "$data_path" \
            --dataset "$dataset" \
            --temperature 1.0 \
            --prompt_style "vanilla"\
            --lora_path "$lora_path" # LoRA adapter
        
    done
done