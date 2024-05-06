
#LR=1e-4
MODE=${1:-"0"}
LR=${2:-"1e-5"}

prompt_continue() {
    while true; do
        read -p "Do you want to continue? [y/n]: " input
        case $input in
            [Yy]* ) break;;
            [Nn]* ) echo "Exiting program."; exit;;
            * ) echo "Invalid input. Please enter y or n.";;
        esac
    done
}
echo "MODE = ${MODE}, LR = ${LR}"
prompt_continue

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

deepspeed --num_gpus=4 --master_port $MASTER_PORT main.py \
    --deepspeed deepspeed.json \
    --do_train \
    --train_file /data/yanghq/datasets/AdvertiseGen/train.json \
    --test_file /data/yanghq/datasets/AdvertiseGen/smalldev.json \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path /data/yanghq/models/YHQ/chatglm-6b \
    --output_dir /data/yanghq/outputs/ft-mode_${MODE}}-lr_${LR} \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --predict_with_generate \
    --max_steps 80 \
    --logging_steps 10 \
    --save_steps 80 \
    --preprocessing_num_workers 16 \
    --learning_rate ${LR} \
    --fp16
#   --pre_seq_len 128 \
#   --ptoken ${MODE} \