PRE_SEQ_LEN=128
LR=2e-2

CUDA_VISIBLE_DEVICES=1 python3 main.py \
    --do_train \
    --train_file /data/yanghq/datasets/AdvertiseGen/train.json \
    --validation_file /data/yanghq/datasets/AdvertiseGen/dev.json \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path /data/yanghq/models/THUDM/chatglm-6b-int4 \
    --output_dir /data/yanghq/outputs/pt-mode0-int4-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 50 \
    --logging_steps 10 \
    --save_steps 50 \
    --learning_rate $LR \
    --quantization_bit 4 \
    --preprocessing_num_workers 16 \
    --pre_seq_len $PRE_SEQ_LEN

