
models=(
'google/gemma-3-4b-it'
'google/gemma-3-12b-it'
)

datasets=(
    'simple_qa_all_science_technology'
)

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        model_name=$(basename "$model")
        data_path="evaluating_LC/datasets/${dataset}.csv"

        PYTHONPATH=lib python3 bin/su_script.py \
            --model_name "$model" \
            --data_path "$data_path" \
            --dataset "$dataset" \
            --temperature 1.0\
            --sampling_num 10
    done
done