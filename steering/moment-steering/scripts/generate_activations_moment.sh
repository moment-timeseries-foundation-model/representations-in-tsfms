#!/bin/bash

dataset_dir="datasets"
device="cpu"
log_file="progress.log"
log_level="DEBUG"
datasets=$(ls $dataset_dir/*.parquet)

echo "Datasets to process:" | tee $log_file
for dataset in $datasets; do
    echo "$(basename $dataset)" | tee -a $log_file
done

echo "-----------------------------------------------" | tee -a $log_file

for dataset in $datasets; do
    dataset_name=$(basename $dataset)
    
    echo "Processing ${dataset_name} on ${device} with logging level ${log_level}..." | tee -a $log_file
    start_time=$(date +%s)
    
    python -m core.moment --dataset "${dataset}" --device "${device}" --log "${log_level}"

    if [ $? -eq 0 ]; then
        echo "Finished processing ${dataset_name}" | tee -a $log_file
    else
        echo "Error processing ${dataset_name}" | tee -a $log_file
    fi

    end_time=$(date +%s)
    time_taken=$((end_time - start_time))
    echo "Time taken for ${dataset_name}: ${time_taken} seconds" | tee -a $log_file
    
    echo "-----------------------------------------------" | tee -a $log_file
done

echo "All datasets processed." | tee -a $log_file