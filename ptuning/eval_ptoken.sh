PRE_SEQ_LEN=128
CHECKPOINT=ptoken-4b-128-2e-2
STEP=3000

CUDA_VISIBLE_DEVICES=1 python3 main.py \
    --do_predict \
    --validation_file /data/yanghq/datasets/AdvertiseGen/smalldev.json \
    --test_file /data/yanghq/datasets/AdvertiseGen/smalldev.json \
    --overwrite_cache \
    --prompt_column content \
    --response_column summary \
    --model_name_or_path /data/yanghq/models/YHQ/chatglm-6b \
    --ptuning_checkpoint /data/yanghq/outputs/$CHECKPOINT/checkpoint-$STEP \
    --output_dir ./output/$CHECKPOINT \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --pre_seq_len $PRE_SEQ_LEN \
    --preprocessing_num_workers 16 \
    --quantization_bit 4 
