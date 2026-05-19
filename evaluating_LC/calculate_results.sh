datasets=(
    4b-st_lc
    4b-st_lc_plus
    4b-st_fine_tuned
    4b-art_lc
    4b-art_lc_plus
    4b-art_fine_tuned
    4b-geo_lc
    4b-geo_lc_plus
    4b-geo_fine_tuned
    4b-simple_qa_lc
    4b-simple_qa_lc_plus
    4b-simple_qa_fine_tuned
    4b-nq_open_lc
    4b-nq_open_lc_plus
    4b-nq_open_fine_tuned
    12b-st_lc
    12b-st_lc_plus
    12b-st_fine_tuned
    12b-art_lc
    12b-art_lc_plus
    12b-art_fine_tuned
    12b-geo_lc
    12b-geo_lc_plus
    12b-geo_fine_tuned
    12b-simple_qa_lc
    12b-simple_qa_lc_plus
    12b-simple_qa_fine_tuned
    12b-nq_open_lc
    12b-nq_open_lc_plus
    12b-nq_open_fine_tuned
)


for dataset in "${datasets[@]}"; do
    data_path="evaluating_LC/outputs/LC_outputs/results/${dataset}_results.csv"
    PYTHONPATH=lib python3 bin/get_scores.py \
        --data_path "$data_path" \
        --dataset "$dataset" \
        --auroc_exclude_not_attempted False \
        --ece_exclude_not_attempted True \
        --accuracy_exclude_not_confident True \
        --ece_bins 10 \
        --output_path "evaluating_LC/outputs/LC_outputs/scores/scores.jsonl"
   
    done
done