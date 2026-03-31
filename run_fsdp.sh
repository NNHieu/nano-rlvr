
NPROC=$1

DATA_PATH=/home/nlp/hnn5071/datasets/mathgraph/data/pathstar_2_10_forward/train/train.parquet
DATA_SIZE=1610 # Train 1600, Val 10
MAX_ITER=1600 # 16 epochs
MODEL_NAME=/home/nlp/hnn5071/models/pathstar/unsloth_qwen2.5_0.5b_sft_ep16_pathstar_2_10_forward/checkpoint-100
RUN_ID=pathstar_2_10-qwen_forward-grpo
SAVE_STEPS=100
TEST_FREQ=50
RUN_DIR=runs
PER_DEVICE_BATCH_SIZE=16

python nano_r1_fsdp.py --nproc $NPROC --model_name $MODEL_NAME --run_id $RUN_ID --save_steps $SAVE_STEPS --test_freq $TEST_FREQ --run_dir $RUN_DIR --data_path $DATA_PATH --data_size $DATA_SIZE --per_device_batch_size $PER_DEVICE_BATCH_SIZE --max_iter $MAX_ITER